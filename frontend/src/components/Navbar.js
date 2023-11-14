import React from 'react';
import {Link} from "react-router-dom";
import '../styles/Navbar.css';

function Navbar() {
    return (
        <div className="Navbar">
            <div className="leftSide">
                <span className="logo">EN POISSON</span>
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
