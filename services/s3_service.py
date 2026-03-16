import boto3
import os
from dotenv import load_dotenv

load_dotenv()


class S3Service:

    def __init__(self):

        self.bucket_name = os.getenv("S3_BUCKET_NAME")

        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME not set")

        self.s3 = boto3.client("s3")

    def upload_file(self, file_obj, filename):

        self.s3.upload_fileobj(
            file_obj,
            self.bucket_name,
            filename
        )

        return f"s3://{self.bucket_name}/{filename}"