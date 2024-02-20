import React, { useState, useEffect } from 'react';
import Chessboard from 'chessboardjsx';
import { Chess } from 'chess.js';

const Analyzed = () => {
  const [fen, setFen] = useState('start');
  const [chess] = useState(new Chess());

  const makeRandomMove = () => {
    const moves = chess.moves();
    if (moves.length === 0) return;

    const randomMove = moves[Math.floor(Math.random() * moves.length)];
    chess.move(randomMove);
    setFen(chess.fen());
  };

  useEffect(() => {
    const interval = setInterval(() => {
      makeRandomMove();
    }, 1000); // Adjust the interval as needed
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
