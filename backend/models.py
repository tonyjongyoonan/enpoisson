from pydantic import BaseModel, EmailStr, StringConstraints
from typing_extensions import Annotated, Optional, Literal
from annotated_types import MaxLen

elo_type = Literal[1100, 1500, 1900]


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
    top_k: int = 3
    elo: elo_type


class Difficulty(BaseModel):
    fen: str
    last_16_moves: Annotated[list[str], MaxLen(16)]
    is_white_move: bool
    elo: elo_type = 1500


class ChessExplanation(BaseModel):
    fen: str
    move: str
    is_white_move: bool
