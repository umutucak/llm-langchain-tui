"""The / commands, one function each, dispatched from the input loop."""

from llmtui.sessions import delete_session, list_sessions, set_session_title


def cmd_help() -> None:
    print("/name -- Name your current conversation and save to database.")
    print("/list -- See all conversations so far.")
    print("/load -- Resume a previous conversation.")
    print("/delete -- Delete a previous conversation.")


def cmd_name(sqlite_connection, agent_thread_config) -> None:
    session_title = input(" > Enter the title of the session: ")
    set_session_title(sqlite_connection, agent_thread_config["configurable"]["thread_id"], session_title)


def cmd_list(sqlite_connection, agent_thread_config) -> None:
    sessions = list_sessions(sqlite_connection)
    for i, (thread_id, title, updated_at) in enumerate(sessions, start=1):
        if thread_id == agent_thread_config["configurable"]["thread_id"]:
            print("[CURRENT] ", end="")
        print(f"{i}. {title or '(untitled)'}  updated {updated_at}")


def cmd_load(sqlite_connection, agent_thread_config) -> None:
    sessions = list_sessions(sqlite_connection)
    for i, (thread_id, title, updated_at) in enumerate(sessions, start=1):
        if thread_id == agent_thread_config["configurable"]["thread_id"]:
            print("[CURRENT] ", end="")
        print(f"{i}. {title or '(untitled)'}  updated {updated_at}")
    selected = int(input(" > Enter index of session to resume: ")) - 1
    agent_thread_config["configurable"]["thread_id"] = sessions[selected][0]


def cmd_delete(sqlite_connection, agent_thread_config) -> None:
    sessions = list_sessions(sqlite_connection)
    current_idx: int = 0
    for i, (thread_id, title, updated_at) in enumerate(sessions, start=1):
        # skip current one
        if thread_id == agent_thread_config["configurable"]["thread_id"]:
            current_idx = i
            print("[CURRENT] ", end="")
        print(f"{i}. {title or '(untitled)'}  updated {updated_at}")
    selected = input(" > Enter indices of sessions to delete separated with white space (1 2 4): ")
    for i in selected.strip().split(' '):
        idx = i.strip()
        if idx.isnumeric() and int(idx) > 0 and int(idx) <= len(sessions) and int(idx) != current_idx:
            idx = int(idx)-1
            # commenting this out for now for testing... surely i wont forget and then get confused
            # confirm = input("Confirm session deletion by typing YES: ")
            confirm = "YES"
            if confirm == "YES":
                delete_session(sqlite_connection, sessions[idx][0])
                print("Delete successful.")
            else:
                print("Delete not confirmed.\n")


def dispatch(user_input: str, sqlite_connection, agent_thread_config) -> None:
    # big ass ugly ass switch case, look away before it turns you to stone
    match user_input[1:]:
        case "help":
            cmd_help()
        case "name":
            cmd_name(sqlite_connection, agent_thread_config)
        case "list":
            cmd_list(sqlite_connection, agent_thread_config)
        case "load":
            cmd_load(sqlite_connection, agent_thread_config)
        case "delete":
            cmd_delete(sqlite_connection, agent_thread_config)
        case _:
            print(f"{user_input[1:]} unrecognized. Try /help to see available commands.")
