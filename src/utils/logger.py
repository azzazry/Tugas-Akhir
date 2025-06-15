log_lines = []

def log_line(line):
    print(line, flush=True)
    log_lines.append(line)

def get_last_log(n=1):
    return log_lines[-n:] if len(log_lines) >= n else log_lines

def clear_log_lines():
    log_lines.clear()
    
def flush_logs(filepath):
    global log_lines
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in log_lines:
            f.write(line + '\n')