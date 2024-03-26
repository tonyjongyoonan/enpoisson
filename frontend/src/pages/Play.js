// TODO: implement checkmate

import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { Chessboard } from 'react-chessboard';
import { Chess } from 'chess.js'; // Corrected import statement
import './Play.css';

const Play = () => {
  const [fen, setFen] = useState('start');
  const chess = useRef(new Chess());
  const location = useLocation();
  const [color, setColor] = useState(null);
  const [turn, setTurn] = useState('white');

  const playRandomMove = () => {
    const moves = chess.current.moves();
    const move = moves[Math.floor(Math.random() * moves.length)];
    chess.current.move(move);
    setFen(chess.current.fen());
    if (chess.current.isCheckmate()) {
      console.log('checkmate');
    } else if (chess.current.isDraw()) {
      console.log('draw');
    }
  };

  const handleColorChange = (newColor) => {
    setColor(newColor);
    if (newColor === 'black') {
      const opening_moves = ['e4', 'd4', 'Nf3']
      // play one out of the three
      const move = opening_moves[Math.floor(Math.random() * opening_moves.length)];
      setTimeout(() => {
        chess.current.move(move);
        setFen(chess.current.fen());
      }, 500);
    }
  };
  
  const getEngineMove = async () => {
    // const model_input_json = { "moves": [] };
    const no_moves = chess.current.history().length;
    // const history = chess.current.history().slice(0, no_moves);
    // const moves_made = moves.slice(0, no_moves); // gets all moves so far
    // for (let i = 0; i < no_moves; i++) {
    //   model_input_json.moves.push({ "fen": chess.current.fen(), "last_16_moves": moves_made.slice(Math.max(0, i - 16), i), "is_white": history[i].color === 'w' });
    // }
    try {
      const response = await fetch("http://localhost:8000/get-human-move", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          fen: chess.current.fen(), 
          last_16_moves: chess.current.history().slice(Math.max(0, no_moves - 16), no_moves),
          is_white_move: turn === 'white' ? false : true // if player is white, then return false (black) since we want model to give black move
        })
      });
      const data = await response.json();
      console.log(data);
      const moves = Object.keys(data);
      const probabilities = Object.values(data);
      const sumProb = probabilities.reduce((a, b) => a + b, 0);
      for (let i = 0; i < probabilities.length; i++) {
        probabilities[i] /= sumProb;
      }
      console.log(moves);
      console.log(probabilities);
      const threshold = Math.random();
      console.log("threshold: " + threshold);
      let runningProb = probabilities[0];
      let selectedMove = null;
      for (let i = 0; i < moves.length; i++) {
        console.log("runningProb: " + runningProb);
        if (runningProb > threshold) {
          selectedMove = moves[i];
          break;
        }
        runningProb += probabilities[i + 1];
      }

      if (selectedMove) {
        chess.current.move(selectedMove);
        setFen(chess.current.fen());
        if (chess.current.isCheckmate()) {
          console.log('checkmate');
        } else if (chess.current.isDraw()) {
          console.log('draw');
        }
      } else {
        console.log('error: no move selected');
      }
    } catch (error) {
      console.log(error);
    }
  }

  // Function to handle piece drop
  const onDrop = (sourceSquare, targetSquare, piece) => {
    try {
      // if (turn !== chess.current.turn()) return;
      let move = chess.current.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q'
      });
      if (move === null) return;
      if (chess.current.isGameOver() || chess.current.isDraw()) return false; // TODO: FIX
      setFen(chess.current.fen());
      if (chess.current.isCheckmate()) {
        console.log('checkmate');
      } else if (chess.current.isDraw()) {
        console.log('draw');
      }

      setTimeout(() => {
        getEngineMove();
      }, 500);
      if (chess.current.isCheckmate()) {
        console.log('checkmate');
        <input type="text" value="Game is over" readOnly />
      } else if (chess.current.isDraw()) {
        console.log('draw');
        <input type="text" value="Game is over" readOnly />
      }
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
              boardOrientation={color}
              boardWidth={560}
              arePiecesDraggable={true}
            />
          </div>

          <div className="game-buttons">
            <div><button className="reset-button" onClick={() => {
              chess.current.reset();
              setFen(chess.current.fen());
              if (color === 'black') {
                handleColorChange('black');
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