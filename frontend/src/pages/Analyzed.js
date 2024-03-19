import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import Chessboard from 'chessboardjsx';
import { Chess } from 'chess.js';
import './Analyzed.css';

const Analyzed = () => {
  const [fen, setFen] = useState('start');
  const chess = useRef(new Chess());
  const moveIndex = useRef(0);
  const moves = useRef([]);
  const location = useLocation();
  const pgn = location.state.pgn;
  const moveIndexToFeedback = useRef({});
  const feedback = "";

  const getPgnMoves = (pgn) => {
    const newChess = new Chess();
    newChess.loadPgn(pgn);
    console.log(newChess.history());
    return newChess.history();
  };

  useEffect(() => {
    moves.current = getPgnMoves(pgn);
    const handleKeydown = (event) => {
      if (event.key === 'ArrowRight' && moveIndex.current < moves.current.length) {
        chess.current.move(moves.current[moveIndex.current]);
        setFen(chess.current.fen());
        moveIndex.current++;

        // if not cached, request for feedback
        if (!moveIndexToFeedback.current[moveIndex.current]) {
          // request for feedback

        }
        feedback = "Feedback";

        // store feedback in hashmap
        moveIndexToFeedback.current[moveIndex.current] = feedback;

      } else if (event.key === 'ArrowLeft' && moveIndex.current > 0) {
        chess.current.undo();
        setFen(chess.current.fen());
        moveIndex.current--;

        // get cached feedback
        feedback = moveIndexToFeedback.current[moveIndex.current];
        
        // display feedback


      }
    };
    window.addEventListener('keydown', handleKeydown);
    return () => window.removeEventListener('keydown', handleKeydown);
  }, [pgn]);

  return (
    <div className="analysis-page-container">
      <div className="chessboard-container">
        <Chessboard position={fen} />
      </div>
      <div className="analysis-text-container">
        <h2>Analysis</h2>
        <p>TonySoTender vs. chmuina</p>
        <p>{"Here is the placeholder feedback--Andrew, when you have a chance to see this, can you please fix the formatting so that it's flexed properly for different sized feedback strings?"}</p>
      </div>
    </div>
  );
};  
export default Analyzed;