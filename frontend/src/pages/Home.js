import "./Home.css";
import React from "react";

function Home() {
  return (
    <div class="grid-container" style={{ backgroundColor: "white" }}>
      <div
        class="box"
        style={{
          width: "100%",
          height: "80%",
          gridColumnEnd: 3,
        }}
      >
        Hi
      </div>

      <div
        class="box"
        style={{
          width: "100%",
          height: "20%",
          backgroundColor: "aliceblue",
          gridColumnStart: 3,
        }}
      >
        Hi
      </div>
    </div>
  );
}

export default Home;
