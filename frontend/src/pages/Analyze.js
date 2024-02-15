import React from 'react'
import { reducer } from "../reducer/reducer";
import { useReducer } from "react";
import { initGameState } from "../constants";
import AppContext from "../contexts/Context";
import Board from "../components/Board/Board";
import Control from "../components/Control/Control";
import Footer from "../components/Footer";
import TakeBack from "../components/Control/bits/TakeBack";
import MovesList from "../components/Control/bits/MovesList";
import FileUpload from '../components/FileUpload';

function Analyze() {
  const [pgn, setPgn] = React.useState('');
  const handleFileUpload = (pgnData) => {
    setPgn(pgnData);
  }
  const [appState, dispatch] = useReducer(reducer, initGameState);
  const providerState = {
    appState,
    dispatch,
  };
  return (
    <AppContext.Provider value={providerState}>
      <div>
      <h1>Analyzer</h1>
      <FileUpload onFileUpload={handleFileUpload} />
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

export default Analyze;
