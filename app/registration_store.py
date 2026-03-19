from __future__ import annotations

from pathlib import Path

from filelock import FileLock

from app.models import RegistrationSessionRecord, RegistrationSubmissionRecord


class RegistrationStore:
    def __init__(self, sessions_dir: Path, submissions_dir: Path) -> None:
        self.sessions_dir = sessions_dir
        self.submissions_dir = submissions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_dir.mkdir(parents=True, exist_ok=True)

    def load_session(self, conversation_id: str) -> RegistrationSessionRecord | None:
        path = self._session_path(conversation_id)
        if not path.exists():
            return None
        try:
            return RegistrationSessionRecord.model_validate_json(path.read_text("utf-8"))
        except (OSError, ValueError):
            return None

    def save_session(self, session: RegistrationSessionRecord) -> RegistrationSessionRecord:
        self._write_record(self._session_path(session.conversation_id), session)
        return session

    def delete_session(self, conversation_id: str) -> None:
        path = self._session_path(conversation_id)
        lock_path = self._lock_path(path)
        with FileLock(str(lock_path)):
            if path.exists():
                path.unlink()

    def list_submissions(self) -> list[RegistrationSubmissionRecord]:
        records: list[RegistrationSubmissionRecord] = []
        for path in self.submissions_dir.glob("*.json"):
            try:
                records.append(RegistrationSubmissionRecord.model_validate_json(path.read_text("utf-8")))
            except (OSError, ValueError):
                continue
        records.sort(key=lambda record: record.submitted_at, reverse=True)
        return records

    def save_submission(self, submission: RegistrationSubmissionRecord) -> RegistrationSubmissionRecord:
        self._write_record(self._submission_path(submission.id), submission)
        return submission

    def _write_record(self, path: Path, record: RegistrationSessionRecord | RegistrationSubmissionRecord) -> None:
        lock_path = self._lock_path(path)
        with FileLock(str(lock_path)):
            path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def _session_path(self, conversation_id: str) -> Path:
        return self.sessions_dir / f"{conversation_id}.json"

    def _submission_path(self, submission_id: str) -> Path:
        return self.submissions_dir / f"{submission_id}.json"

    @staticmethod
    def _lock_path(path: Path) -> Path:
        return path.with_suffix(f"{path.suffix}.lock")
