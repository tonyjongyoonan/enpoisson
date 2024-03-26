import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Link } from "react-router-dom";
import "./Navbar.css";

function Navbar() {
  const [currentUser, setCurrentUser] = useState(null);

  return (
    <div className="Navbar">
      <div className="leftSide">
        <span className="logo" style={{ fontSize: 23 }}>
          EN POISSON
        </span>
        {currentUser && <span>{currentUser}</span>}
      </div>
      <div className="rightSide">
        <Link to="/">Home</Link>
        <Link to="/play">Play</Link>
        <Link to="/analyze">Analyze</Link>
        <Link to="/team">Team</Link>
      </div>
    </div>
  );
}

export default Navbar;
