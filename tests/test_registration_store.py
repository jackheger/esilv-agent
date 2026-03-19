from app.models import RegistrationAnswersRecord, RegistrationSessionRecord, RegistrationSubmissionRecord
from app.registration_store import RegistrationStore


def make_store(tmp_path):
    return RegistrationStore(
        tmp_path / "registrations" / "sessions",
        tmp_path / "registrations" / "submissions",
    )


def test_registration_store_persists_sessions(tmp_path):
    store = make_store(tmp_path)
    session = store.save_session(
        RegistrationSessionRecord(
            conversation_id="conversation-1",
            current_field="email",
            language="en",
            answers=RegistrationAnswersRecord(full_name="Ada Lovelace"),
        )
    )

    loaded = store.load_session("conversation-1")

    assert session.conversation_id == "conversation-1"
    assert loaded is not None
    assert loaded.current_field == "email"
    assert loaded.answers.full_name == "Ada Lovelace"


def test_registration_store_lists_newest_submissions_first(tmp_path):
    store = make_store(tmp_path)
    store.save_submission(
        RegistrationSubmissionRecord(
            id="older",
            conversation_id="conversation-1",
            answers=RegistrationAnswersRecord(full_name="Older Student"),
            submitted_at="2026-03-18T00:00:00+00:00",
        )
    )
    store.save_submission(
        RegistrationSubmissionRecord(
            id="newer",
            conversation_id="conversation-2",
            answers=RegistrationAnswersRecord(full_name="Newer Student"),
            submitted_at="2026-03-19T00:00:00+00:00",
        )
    )

    records = store.list_submissions()

    assert [record.id for record in records] == ["newer", "older"]


def test_registration_store_ignores_malformed_files(tmp_path):
    store = make_store(tmp_path)
    (store.submissions_dir / "broken.json").write_text("{not valid", encoding="utf-8")

    records = store.list_submissions()

    assert records == []
