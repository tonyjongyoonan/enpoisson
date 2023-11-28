import "./Home.css";
import React from "react";

function Home() {
  const gold = "#ffc703";
  return (
    <div class="grid-container" style={{ backgroundColor: "white" }}>
      <div
        class="box"
        style={{
          width: "100%",
          height: "80%",
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
        class="box"
        style={{
          width: "100%",
          height: "20%",
          backgroundColor: "aliceblue",
          gridColumnStart: 2,
        }}
      >
        Hi
      </div>
    </div>
  );
}

export default Home;
