from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

users = {}


class User(BaseModel):
    username: str
    password: str
    email: str


@app.post("/login")
def validate_login(username: str, password: str):
    # FIXME: this we are essentially storing passwords in plain text; use a salt, and a hash
    # or sign in with Google
    if users[username] and users[username].password == password:
        return {"message": f"OK"}
    else:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/account")
def create_account(user: User):
    if (
        user.username is not None
        and user.password is not None
        and user.email is not None
        and user.username != ""
        and user.username not in users
    ):
        username = user.username
        users[username] = user
        return {"message": f"Succesfully created account for {username}."}
    else:
        raise HTTPException(status_code=409, detail="Bad request.")


# @app.get("/profile/{username}")
# def get_bender(name: str):
#     if name not in benders:
#         raise HTTPException(status_code=404, detail="bender not found.")
#     else:
#         return {"bender": name, "element": benders[name]}
