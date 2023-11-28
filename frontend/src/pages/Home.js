import PlusButton from "../components/PlusButton";
import "./Home.css";
import React from "react";

function Home() {
  const gold = "#ffc703";
  return (
    <div class="grid-container">
      <div
        class="home-widget-box"
        style={{
          gridColumnEnd: 2,
          background: "#3f3f3f",
        }}
      >
        <h6 style={{ color: gold, fontWeight: 500 }}>En Poisson</h6>
        <h1 style={{ color: gold }}>
          Perfect Chess <br /> with the premier engine
        </h1>
        <p style={{ color: gold, alignItems: "end" }}>
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
        <p>Learn from your grames with clear explanations</p>
        <h2>Analysis</h2>
        <PlusButton url={"/analysis"} />
      </div>
      <div
        class="home-widget-box"
        style={{
          backgroundColor: "orchid",
          gridColumnStart: 3,
          gridRowStart: 3,
        }}
      >
        <p>Interact with friends & community!</p>
        <h2>Social</h2>
        <PlusButton url={"/social"} />
      </div>
    </div>
  );
}

export default Home;
