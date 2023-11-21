import datetime

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    ORANGE = '\033[33m'
    RESULT =  '\033[94m'
    WARNING = '\033[93m'
    DEBUGGING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def now():
    return datetime.datetime.now().strftime("%H:%M:%S")

def error(buffer):
    custom_print("ERROR", buffer, colors.FAIL)
    
def warning(buffer):
    custom_print("WARNING", buffer, colors.ORANGE)

def debug(buffer):
    custom_print("DEBUG", buffer, colors.DEBUGGING)
    
def info(buffer):
    custom_print("INFO", buffer, colors.RESULT)

def custom_print(label, buffer, color):
    date = now()
    prefix = f'[{colors.BOLD}{color}{label}{colors.ENDC}]'
    print(f'{date} {prefix} {str(buffer)}\n')

if __name__ == "__main__":
    error("main")
    warning("warning")
    debug("debug")
    info("info")