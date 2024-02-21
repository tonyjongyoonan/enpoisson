import React, { useState, useEffect } from 'react';
import Chessboard from 'chessboardjsx';
import { Chess } from 'chess.js';
import { useLocation } from 'react-router-dom';

const Analyzed = () => {
  const location = useLocation();
  const { pgn } = location.state;
  const [fen, setFen] = useState('start');
  const [chess] = useState(new Chess());

  const getPgnMoves = (pgn) => {
    const tempChess = new Chess()
    tempChess.loadPgn(pgn);
    return tempChess.history();
  };

  const makeMove = (moves, moveNo) => {
    chess.move(moves[moveNo]);
    console.log(chess.fen());
    console.log(chess.ascii());
    console.log(moves[moveNo]);
    console.log(moves.length);
    setFen(chess.fen());
  };

  useEffect(() => {
    const moves = getPgnMoves(pgn);
    let moveNo = 0;
    const interval = setInterval(() => {
      if (moveNo < moves.length) {
        makeMove(moves, moveNo);
        moveNo++;
      } else {
        clearInterval(interval);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <Chessboard position={fen} />
      {/* Additional analyzed game content can be added here */}
    </div>
  );
};

export default Analyzed;
