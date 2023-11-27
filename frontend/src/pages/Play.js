import React from "react";
import { reducer } from "../reducer/reducer";
import { useReducer } from "react";
import { initGameState } from "../constants";
import AppContext from "../contexts/Context";
import Board from "../components/Board/Board";
import Control from "../components/Control/Control";
import Footer from "../components/Footer";
import Navbar from "../components/Navbar";
import TakeBack from "../components/Control/bits/TakeBack";
import MovesList from "../components/Control/bits/MovesList";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";

function Play() {
  const [appState, dispatch] = useReducer(reducer, initGameState);
  const providerState = {
    appState,
    dispatch,
  };
  return (
    <AppContext.Provider value={providerState}>
      <div className="App">
        <Navbar />
        <Board />
        <Control>
          <MovesList />
          <TakeBack />
        </Control>
        <Footer />
      </div>
    </AppContext.Provider>
  );
}

export default Play;
