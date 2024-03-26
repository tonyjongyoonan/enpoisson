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
  const [color, setColor] = useState(null);
  const [turn, setTurn] = useState('white');

  const playRandomMove = () => {
    const moves = chess.current.moves();
    const move = moves[Math.floor(Math.random() * moves.length)];
    chess.current.move(move);
    setFen(chess.current.fen());
  };

  const handleColorChange = (newColor) => {
    setColor(newColor);
    if (newColor === 'black') {
      // have a 0.1 second timer, then play a random move
      setTimeout(() => {
        playRandomMove();
      }, 1250);
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
          last_16_moves: chess.current.history().slice(Math.max(0, no_moves - 16), no_moves),
          is_white: chess.current.turn() === 'w'
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
      // getEngineMove();
      setTimeout(() => {
        playRandomMove();
      }, 500);

    } catch (error) {
      console.log(error);
      return null;
    }
  };

  return (
    <div className="play-page-container">
      {!color && (<div className="buttons">
        <div>
          <button className="white-button" onClick={() => handleColorChange('white')}>play as white</button>
        </div>
        <div>
          <button className="black-button" onClick={() => handleColorChange('black')}>play as black</button>
        </div>
      </div>)}

      {color && (
        <div className="game-container">
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
          <div className="game-buttons">
            <div><button className="reset-button" onClick={() => {
              chess.current.reset(); 
              setFen(chess.current.fen());
              // TODO: replace with model move
              if (color === 'black') {
                setTimeout(() => {
                  playRandomMove();
                }, 500);
              }
            }}>reset</button></div>
            <div><button className="undo-button" onClick={() => {
                  setTimeout(() => {
                    chess.current.undo();
                    setFen(chess.current.fen());
                  }, 200); // buffer time looks appropriate
                  setTimeout(() => {
                    chess.current.undo();
                    setFen(chess.current.fen());
                  }, 600); // undo twice to undo model move as well
                }}>undo</button></div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Play;