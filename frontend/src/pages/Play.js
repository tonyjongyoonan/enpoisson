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
  
  const getEngineMove = async () => {
    // const model_input_json = { "moves": [] };
    const no_moves = chess.current.history().length;
    // const history = chess.current.history({ verbose: true }).slice(0, no_moves);
    // const moves_made = moves.slice(0, no_moves); // gets all moves so far
    // for (let i = 0; i < no_moves; i++) {
    //   model_input_json.moves.push({ "fen": chess.current.fen(), "last_16_moves": moves_made.slice(Math.max(0, i - 16), i), "is_white": history[i].color === 'w' });
    // }
    const fen = chess.current.fen();
    const last_16_moves = chess.current.history().slice(Math.max(0, no_moves - 16), no_moves);
    console.log({ fen, last_16_moves });
    try {
      const response = await fetch("http://localhost:8000/get-human-move", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          fen: chess.current.fen(), 
          last_16_moves: chess.current.history().slice(Math.max(0, no_moves - 16), no_moves)
        })
      });
      const data = await response.json();
      console.log(data);
      // Process the engine move data
    } catch (error) {
      console.log(error);
    }
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
      if (move === null) return;
      if (chess.current.isGameOver() || chess.current.isDraw()) return false;
      setFen(chess.current.fen());

      // find engine move
      getEngineMove();

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