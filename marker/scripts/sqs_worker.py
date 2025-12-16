import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click

try:
    import boto3  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("boto3 is required to run sqs_worker") from exc

from marker.config.parser import ConfigParser
from marker.models import create_model_dict
from marker.output import save_output
from marker.scripts.render_from_json import render_json_document


def _load_message_body(msg: Dict[str, Any]) -> Dict[str, Any]:
    body = msg.get("Body")
    if not body:
        return {}
    try:
        return json.loads(body)
    except Exception:
        # Allow plain text messages; treat as S3 key (advanced users can override)
        return {"input_key": body}


def _make_job_dir(base_dir: str, job_id: str) -> str:
    d = os.path.join(base_dir, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _upload_dir_to_s3(s3, local_dir: str, bucket: str, prefix: str):
    prefix = prefix.lstrip("/").rstrip("/")
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, start=local_dir)
            key = f"{prefix}/{rel}".lstrip("/")
            s3.upload_file(local_path, bucket, key)


@click.command(help="Poll SQS for S3 conversion jobs and process with Marker.")
@click.option("--queue-url", required=True, help="SQS queue URL to poll")
@click.option("--region", default=None, help="AWS region (defaults to env/instance)")
@click.option("--input-bucket", default=None, help="Default S3 input bucket (if not in message)")
@click.option("--output-bucket", default=None, help="Default S3 output bucket (if not in message)")
@click.option(
    "--output-prefix",
    default="marker-results",
    show_default=True,
    help="Default output prefix in output bucket",
)
@click.option(
    "--work-dir",
    default="/tmp/marker_jobs",
    show_default=True,
    help="Local working directory on instance",
)
@click.option(
    "--wait-time",
    default=20,
    show_default=True,
    help="SQS long polling wait time (seconds, max 20)",
)
@click.option(
    "--visibility-timeout",
    default=900,
    show_default=True,
    help="Visibility timeout in seconds (should exceed worst-case processing time)",
)
@click.option(
    "--delete-on-receive",
    is_flag=True,
    default=False,
    show_default=True,
    help="Delete the SQS message immediately after receiving/validating it (NOT recommended, but prevents retries).",
)
def sqs_worker_cli(
    queue_url: str,
    region: Optional[str],
    input_bucket: Optional[str],
    output_bucket: Optional[str],
    output_prefix: str,
    work_dir: str,
    wait_time: int,
    visibility_timeout: int,
    delete_on_receive: bool,
):
    """
    Expected SQS message body JSON (minimum):

    {
      "input_bucket": "my-input-bucket",
      "input_key": "uploads/doc.pdf",
      "output_bucket": "my-output-bucket",
      "output_prefix": "marker-results"
    }

    If output_bucket/prefix not provided in message, defaults come from CLI flags.
    """

    os.makedirs(work_dir, exist_ok=True)

    session = boto3.session.Session(region_name=region) if region else boto3.session.Session()
    sqs = session.client("sqs")
    s3 = session.client("s3")

    click.echo("Loading Marker models (first run can take a while)...")
    models = create_model_dict()
    click.echo("Models loaded.")

    while True:
        resp = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=max(0, min(20, int(wait_time))),
            VisibilityTimeout=int(visibility_timeout),
        )

        msgs = resp.get("Messages", [])
        if not msgs:
            # No work
            continue

        msg = msgs[0]
        receipt = msg["ReceiptHandle"]
        message_id = msg.get("MessageId", str(int(time.time())))

        body = _load_message_body(msg)
        in_bucket = body.get("input_bucket") or input_bucket
        in_key = body.get("input_key")
        out_bucket = body.get("output_bucket") or output_bucket
        out_prefix = body.get("output_prefix") or output_prefix

        if not in_bucket or not in_key or not out_bucket:
            click.echo(f"Invalid message (missing buckets/keys): {body}")
            # Delete poison pill to avoid infinite retries
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
            continue

        job_id = body.get("job_id") or message_id
        job_dir = _make_job_dir(work_dir, job_id)
        filename = os.path.basename(in_key) or "document"
        local_input = os.path.join(job_dir, filename)

        try:
            if delete_on_receive:
                # NOTE: This trades reliability for simplicity: if the instance crashes mid-job,
                # the message is already gone and the job will not be retried.
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                receipt = None
                click.echo(f"[{job_id}] Deleted message on receive (no retries).")

            click.echo(f"[{job_id}] Downloading s3://{in_bucket}/{in_key}")
            s3.download_file(in_bucket, in_key, local_input)

            options: Dict[str, Any] = {
                "filepath": local_input,
                "output_format": "json",
                "output_dir": job_dir,
                "layout_batch_size": 1,
                "detection_batch_size": 1,
                "table_rec_batch_size": 1,
                "ocr_error_batch_size": 1,
                "recognition_batch_size": 4,
                "equation_batch_size": 1,
            }

            config_parser = ConfigParser(options)
            config_dict = config_parser.generate_config_dict()
            config_dict["disable_tqdm"] = True

            converter_cls = config_parser.get_converter_cls()
            converter = converter_cls(
                config=config_dict,
                artifact_dict=models,
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
                llm_service=config_parser.get_llm_service(),
            )

            click.echo(f"[{job_id}] Processing with Marker...")
            rendered = converter(local_input)
            base_name = config_parser.get_base_filename(local_input)
            save_output(rendered, job_dir, base_name)

            # Render "our" HTML from the Marker JSON output (stable filename for admin panel).
            json_path = Path(job_dir) / f"{base_name}.json"
            index_html_path = Path(job_dir) / "index.html"
            if json_path.exists():
                render_json_document(json_path, output_path=index_html_path)

            # Upload everything in job_dir to S3 under output_prefix/job_id/
            out_root = f"{out_prefix.rstrip('/')}/{job_id}".strip("/")
            click.echo(f"[{job_id}] Uploading outputs to s3://{out_bucket}/{out_root}/")
            _upload_dir_to_s3(s3, job_dir, out_bucket, out_root)

            # Also upload stable names so the admin panel can fetch by job_id without guessing base_name.
            # (Keeps the original {base_name}.json too.)
            if json_path.exists():
                s3.upload_file(str(json_path), out_bucket, f"{out_root}/job.json")
            if index_html_path.exists():
                s3.upload_file(str(index_html_path), out_bucket, f"{out_root}/index.html")

            # Delete message only on success (unless already deleted on receive)
            if receipt is not None:
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
            click.echo(f"[{job_id}] Done.")

        except Exception as exc:
            click.echo(f"[{job_id}] ERROR: {exc}")
            # Do not delete message; it will retry after visibility timeout
            # You can send to DLQ by configuring redrive policy on the queue.
            continue


if __name__ == "__main__":
    sqs_worker_cli()

