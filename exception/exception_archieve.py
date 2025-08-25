import sys
import traceback
class DocumentPortalException(Exception):
    def __init__(self, error_message, error_details):
        _, _, exc_tb = error_details.exc_info()
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.lineno = exc_tb.tb_lineno
        self.error_message = str(error_message)
        self.traceback_str = ''.join(traceback.format_exception(*error_details.exc_info()))

    def __str__(self):
        return f"""
        Error in [{self.file_name}] at line [{self.lineno}]
        Message: {self.error_message}
        Traceback:
        {self.traceback_str}
        """

if __name__ == "__main__":
    try:
        a = 1 / 0  # deliberate error
    except Exception as e:
        app_exc = DocumentPortalException(e, sys)
        #logger.error(app_exc)  # log it to file
        raise app_exc  # propagate with full traceback
    # try:
    #     a = int("abc")  # ValueError (inbuilt)
    # except ValueError as e:
    #     raise DocumentPortalException("Failed while processing document", e)