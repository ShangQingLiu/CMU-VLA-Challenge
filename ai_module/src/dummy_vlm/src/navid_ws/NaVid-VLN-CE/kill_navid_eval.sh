ps aux | grep 'run.py' | grep 'navid'   | awk '{print $2}' | xargs kill
