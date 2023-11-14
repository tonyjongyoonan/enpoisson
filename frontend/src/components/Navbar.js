import React from 'react';
import Logo from '../assets/pawn.png';
import {Link} from "react-router-dom";
import '../styles/Navbar.css';

function Navbar() {
    return (
        <div className="Navbar">
            <div className="leftSide">
                <img src={Logo} alt="Logo" />
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
