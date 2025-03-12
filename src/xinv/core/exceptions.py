## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

class XinvIllposedError(Exception):
    """Exception raised when Cholesky decomposition is ill-posed 

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, linalgmessage):
        self.message = f"Cholesky decomposition failed: {linalgmessage}"
        super().__init__(self.message)

