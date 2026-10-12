"""Original offline worker, used by the optional static scan entry point."""
import json
import logging
import os
import socket


def offline(*args, **kwargs):
    raise OSError("Network disabled for reproducible SKILL.md-only static scanning")


def configure_offline():
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    os.environ["SKILLSPECTOR_LOG_LEVEL"] = "ERROR"
    socket.create_connection = offline
    socket.getaddrinfo = offline
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    logging.basicConfig(level=logging.ERROR)


def scanner_worker(connection, skill_path):
    configure_offline()
    from skillspector.graph import graph
    connection.send({"ready": True})
    while connection.recv():
        try:
            result = graph.invoke({"input_path": skill_path, "output_format": "json",
                "use_llm": False, "llm_requested": False, "source_local_only": True},
                config={"max_concurrency": 1})
            connection.send(json.loads(result["report_body"]))
        except Exception as error:
            connection.send({"scanner_error": {"type": type(error).__name__, "message": str(error)}})


def unavailable(error):
    # This is a driver failure record, not a fabricated native score or safe verdict.
    return {"risk_assessment": {"score": None, "severity": "UNAVAILABLE", "recommendation": "UNAVAILABLE"},
        "issues": [], "analysis_completeness": {"is_complete": False, "status": "failed",
            "execution_successful": False, "ledger_exceptions": [{"reason_code": "scanner_" + error["type"],
                "message": error["message"]}]}, "metadata": {}, "scanner_error": error}
