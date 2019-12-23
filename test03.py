import logging

logging.basicConfig(level=logging.INFO,filename='test.log')
logging.error("出现了错误")
logging.info("打印信息")
logging.warning("警告信息")