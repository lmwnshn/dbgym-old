import threading
from queue import Queue

import pandas as pd
import psutil
from sqlalchemy import create_engine
from sqlalchemy.pool import SingletonThreadPool
from pathlib import Path

import time

from dbgym.util.sql import substitute


def submission_worker(workload_db, prepare_queue, work_queue, prepare_event, done_event):
    engine = create_engine(f"sqlite:///{workload_db}", poolclass=SingletonThreadPool, pool_size=psutil.cpu_count())
    # TODO(WAN): Figure out the semantics of elapsed_s. Do we care? Or replay goes brr?
    max_query_num = engine.execute("select max(query_num) from workload").fetchone()[0]
    cur_query_num = 0
    tick_query_num = 500000

    sql = "select id, template from query_template"
    results = engine.execute(sql).fetchall()
    for result in results:
        prepare_queue.put(result)
    prepare_event.set()

    sql = (
        "select distinct(template_id) "
        "from workload where params_id is null "
        "order by template_id"
    )
    templates_without_params = engine.execute(sql).fetchall()
    sql = (
        "select distinct(template_id) "
        "from workload where params_id is not null "
        "order by template_id"
    )
    templates_with_params = engine.execute(sql).fetchall()

    while True:
        all_results = []
        for template_id, *_ in templates_without_params:
            sql = (
                "select w.query_num, w.template_id "
                f"from workload w where "
                f"{cur_query_num} <= query_num and "
                f"query_num < {cur_query_num + tick_query_num} and "
                f"w.template_id = {template_id} "
                "order by w.query_num"
            )
            results = engine.execute(sql).fetchall()
            results = [(qnum, template_id, None) for qnum, template_id in results]
            all_results.extend(results)
        for template_id, *_ in templates_with_params:
            sql = (
                "select w.query_num, w.template_id, p.* "
                f"from workload w, template_{template_id}_params p where "
                f"{cur_query_num} <= query_num and "
                f"query_num < {cur_query_num + tick_query_num} and "
                "w.params_id = p.id and "
                f"w.template_id = {template_id} "
                "order by w.query_num"
            )
            results = engine.execute(sql).fetchall()
            results = [(qnum, template_id, params) for qnum, template_id, params_id, *params in results]
            all_results.extend(results)
        all_results.sort(key=lambda x: x[0])

        for query_num, template_id, params in all_results:
            work_queue.put((template_id, params))

        cur_query_num = cur_query_num + tick_query_num
        if cur_query_num >= max_query_num:
            break
    done_event.set()


workload_db = Path("./artifact/gym/workload.db")

prepare_queue = Queue()
work_queue = Queue()
prepare_event = threading.Event()
done_event = threading.Event()
threading.Thread(target=submission_worker, args=(workload_db, prepare_queue, work_queue, prepare_event, done_event), daemon=True).start()

while not prepare_event.wait(1):
    pass
while not prepare_queue.empty():
    template_id, template = prepare_queue.get()
    sql = f"PREPARE work_{template_id} AS {template}"
    print(sql)
    prepare_queue.task_done()

start_time = time.time()

def format_params(params):
    if params is None:
        return ""
    # Each param is quoted.
    return "(" + ",".join(params) + ")"

tasks = 0
while True:
    if done_event.is_set() and work_queue.empty():
        break
    tasks += 1
    template_id, params = work_queue.get()
    sql = f"EXECUTE work_{template_id}{format_params(params)};"
    print(sql)
    work_queue.task_done()
    if tasks % 5000 == 0:
        print(f"Finished {tasks}.")

end_time = time.time()
print(end_time - start_time, " elapsed time")