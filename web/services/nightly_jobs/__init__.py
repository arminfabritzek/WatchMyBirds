"""Concrete nightly jobs registered with web.services.nightly_job_hub."""

from web.services.nightly_jobs.aesthetic_tagger_job import AestheticTaggerJob
from web.services.nightly_jobs.retention_job import RetentionJob
from web.services.nightly_jobs.sharpness_job import SharpnessJob

__all__ = ["AestheticTaggerJob", "RetentionJob", "SharpnessJob"]
