import React from "react";
import { Link } from "react-router-dom";
import "./Navbar.css";

function Navbar() {
  return (
    <div className="Navbar">
      <div className="leftSide">
        <span className="logo" style={{ fontSize: 23 }}>
          EN POISSON
        </span>
      </div>
      <div className="rightSide">
        <Link to="/">Home</Link>
        <Link to="/play">Play</Link>
        <Link to="/about">About</Link>
        <Link to="/login">Login</Link>
      </div>
    </div>
  );
}

export default Navbar;
