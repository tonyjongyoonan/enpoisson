import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import Chessboard from 'chessboardjsx';
import { Chess } from 'chess.js';
import './Play.css';

const Play = () => {
  const [fen, setFen] = useState('start');
  const chess = useRef(new Chess());
  const moveIndex = useRef(0);
  const moves = useRef([]);
  const location = useLocation();
  const moveIndexToFeedback = useRef({});
  const feedback = "";


  return (
    <div className="play-page-container">
      <div className="chessboard-container">
        <Chessboard position={fen} />
      </div>
    </div>
  );
};  
export default Play;