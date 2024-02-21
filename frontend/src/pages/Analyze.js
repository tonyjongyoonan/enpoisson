import React from 'react'
import { reducer } from "../reducer/reducer";
import { useReducer } from "react";
import { initGameState } from "../constants";
import AppContext from "../contexts/Context";
import Footer from "../components/Footer";
import FileUpload from '../components/FileUpload';
import './Analyze.css';
import { useNavigate } from 'react-router-dom';

function Analyze() {
  const [appState, dispatch] = useReducer(reducer, initGameState);
  const providerState = {
    appState,
    dispatch,
  };

  const navigate = useNavigate();
  const handleFileUpload = (pgnData) => {
    navigate('/analyzed', { state: { pgn: pgnData } });
  };

  return (
    <AppContext.Provider value={providerState}>
      <div className="analyzer-container">
        <div className="file-upload">
          <FileUpload onFileUpload={handleFileUpload} />
          </div>
        <Footer />
      </div>
    </AppContext.Provider>
  );
}

export default Analyze;
