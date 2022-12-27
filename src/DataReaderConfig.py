from unicodedata import numeric


def noRejectionBasedOnPrior(self):
    return False
    
def noRejectionBasedOnSimulation(self):
    return False

    
def rejectLowPrior(self, prior: dict, lower_bound: numeric):
    if any(x < lower_bound for x in prior.values()):
        return True