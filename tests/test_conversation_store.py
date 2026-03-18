from app.conversation_store import ConversationStore
from app.models import MessageRecord


def test_conversation_store_persists_messages_and_titles(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()

    store.append_message(conversation.id, MessageRecord(role="user", content="Tell me about ESILV admissions"))
    store.append_message(conversation.id, MessageRecord(role="assistant", content="Admissions details"))

    loaded = store.load(conversation.id)
    assert loaded.title == "Tell me about ESILV admissions"
    assert [message.role for message in loaded.messages] == ["user", "assistant"]


def test_conversation_store_lists_latest_first_and_can_delete(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    first = store.create()
    second = store.create()

    store.append_message(first.id, MessageRecord(role="user", content="First conversation"))
    store.append_message(second.id, MessageRecord(role="user", content="Second conversation"))

    listed = store.list()
    assert [record.id for record in listed][:2] == [second.id, first.id]

    store.delete(first.id)
    remaining_ids = [record.id for record in store.list()]
    assert first.id not in remaining_ids
