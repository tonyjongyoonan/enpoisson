from pydantic import BaseModel, EmailStr, StringConstraints
from typing_extensions import Annotated


class UserCreate(BaseModel):
    username: Annotated[  # type: ignore
        str,
        StringConstraints(
            strip_whitespace=True, to_upper=False, pattern=r"^[a-zA-Z0-9]+$"
        ),
    ]
    password: str
    email: EmailStr


class UserLogin(BaseModel):
    username: str
    password: str


class ChessPosition(BaseModel):
    fen: str
    # is_white_move: bool
