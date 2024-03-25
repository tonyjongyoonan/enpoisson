from pydantic import BaseModel, EmailStr, StringConstraints
from typing_extensions import Annotated
from annotated_types import MaxLen


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
    last_16_moves: Annotated[list[str], MaxLen(16)]
    is_white_move: bool
