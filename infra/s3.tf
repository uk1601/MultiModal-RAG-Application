resource "aws_s3_bucket" "assignment_bucket" {
  bucket = "bigdata-f24-t1-assignment"

  tags = {
    Name        = "bigdata_f24_t1_assignment"
    Environment = "Dev"
  }
}