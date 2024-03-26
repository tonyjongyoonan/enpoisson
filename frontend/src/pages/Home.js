import PlusButton from "../components/PlusButton";
import "./Home.css";
import React, { useEffect, useState } from 'react';
import axios from 'axios';

function Home() {
  const gold = "#ffc703";
  const [currentUser, setCurrentUser] = useState(null);

  return (
    <div class="grid-container">
      <div
        class="home-widget-box"
        style={{
          gridColumnEnd: 2,
          background: "#303030",
        }}
      >
        <h6 style={{ color: gold, fontWeight: 400 }}>EN POISSON</h6>
        <h1 style={{ color: gold, fontWeight: 350, lineHeight: "1.1", marginBottom: "80px" }}>
          {currentUser ? 'Welcome, ${currentUser}' : 'Log in to play!'}
        </h1>
        <p style={{ color: gold, alignItems: "end", marginBottom: "20px", fontWeight: 350 }}>
          Learn chess better with our new innovative chess engine, recommending
          you moves you'll actually find!
        </p>
      </div>

      <div
        class="home-widget-box"
        style={{
          backgroundColor: "azure",
          gridColumnStart: 2,
          gridRowStart: 2,
          gridRowEnd: 3,
        }}
      >
        <PlusButton url={"/play"} />
        <p style={{ color: "mediumvioletred" }}>
          Play against our innovative, human-like chess engine through a
          learning mode
        </p>
        <h2 style={{ color: "mediumvioletred" }}>Play Engine</h2>
      </div>

      <div
        class="home-widget-box"
        style={{
          backgroundColor: "royalblue",
          gridColumnStart: 2,
          gridColumnEnd: 3,
          gridRowStart: 3,
        }}
      >
        <p>Learn via AI-fueled explanations</p>
        <h2>Analysis</h2>
        <PlusButton url={"/analyze"} />
      </div>
      <div
        class="home-widget-box"
        style={{
          backgroundColor: "orchid",
          gridColumnStart: 3,
          gridRowStart: 3,
        }}
      >
        <p>Create an account to play!</p>
        <h2>Register</h2>
        <PlusButton url={"/account"} />
      </div>
    </div>
  );
}

export default Home;
