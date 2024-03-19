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
        const feedback = "Feedback";

        // store feedback in hashmap
        moveIndexToFeedback.current[moveIndex.current] = feedback;

      } else if (event.key === 'ArrowLeft' && moveIndex.current > 0) {
        chess.current.undo();
        setFen(chess.current.fen());
        moveIndex.current--;

        // get cached feedback
        const feedback = moveIndexToFeedback.current[moveIndex.current];
        
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
        <p>{"The game starts with the Philidor Defense, which begins after 1.e4 e5 2.Nf3 d6. This is a solid opening choice from Black, aiming for a strong pawn structure and control over the center.\
        White opts for a restrained setup with 4.h3, preventing any piece from pinning the knight on f3. Black's 7...Bxc4 followed by 8...Nd4 indicates an attempt to simplify the position, but White recaptures with the knight, maintaining a strong pawn structure.\
        After White's queen and rook coordinate effectively to pressure Black's position, Black resigns on move 65, facing the loss of more material and an untenable position."}</p>
      </div>
    </div>
  );
};  
export default Analyzed;