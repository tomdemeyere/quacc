from quacc import SETTINGS

def resolve(func):
    """To be used only with a very good reason. This is not good practice."""
    if SETTINGS.WORKFLOW_ENGINE == "parsl":
        return func.result()
    else:
        return func
