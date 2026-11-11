import ast
src=open("outputs/q_main_20261017/main_suite.py").read(); ast.parse(src)
ns={}; exec(src.split("if __name__")[0], ns)
print("parsed", len(ns["RUNS"]), "remaining_s", ns["remaining"](), "secs", ns["_secs"]("1-02:03:04"), ns["_secs"]("11:50"), "alive_self", ns["alive"]("57798395"), "alive_dead", ns["alive"]("57648262"))
