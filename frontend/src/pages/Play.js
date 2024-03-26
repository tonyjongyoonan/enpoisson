// TODO: implement checkmate

import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { Chessboard } from 'react-chessboard';
import { Chess } from 'chess.js'; // Corrected import statement
import './Play.css';

const Play = () => {
  const [fen, setFen] = useState('start');
  const chess = useRef(new Chess());
  const [index, setIndex] = useState(0);
  const [moves, setMoves] = useState([]);
  const location = useLocation();
  const [selected, setSelected] = useState({ value: 'game', label: 'Played move'});
  const [arrows, setArrows] = useState([]);
  const [color, setColor] = useState('white');
  const [turn, setTurn] = useState('white');

  const handleColorChange = (newColor) => {
    setColor(newColor);
    if (newColor === 'black') {
      // run the first move
    }
  };
  
  const getEngineMove =  {

  }

  // Function to handle piece drop
  const onDrop = (sourceSquare, targetSquare, piece) => {
    try {
      
      console.log(chess.current.turn());
      // if (turn !== chess.current.turn()) return;
      let move = chess.current.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q'
      });
      console.log(move);
      if (move === null) return;
      if (chess.current.isGameOver() || chess.current.isDraw()) return false;
      setFen(chess.current.fen());

      // find engine move

    } catch (error) {
      console.log(error);
      return null;
    }
  };

  return (
    <div className="play-page-container">
      <div className="chessboard-container">
        <Chessboard
          position={fen}
          onPieceDrop={onDrop}
          onPieceClick={(square) => console.log({ square })}
          boardOrientation={color}
          boardWidth={560}
          arePiecesDraggable={true}
        />
      </div>
      <div className="color-choice-container">
        <button onClick={() => handleColorChange('white')}>Play as White</button>
        <button onClick={() => handleColorChange('black')}>Play as Black</button>
      </div>
      <div className="play-button-container">
      <button onClick={() => {
          chess.current.reset();
          setFen(chess.current.fen());
        }}>reset</button>
      <button onClick={() => {
          chess.current.undo();
          chess.current.undo(); // undo twice to undo model move as well
          // TODO: make sure to error check for when there's only one move played
          setFen(chess.current.fen());
        }}>undo</button>
      </div>
    </div>
  );
};  

export default Play;