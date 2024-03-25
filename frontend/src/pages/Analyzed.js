import React, { useState, useEffect, useRef, Fragment } from 'react';
import Select from 'react-select';
import { useLocation } from 'react-router-dom';
import { Chessboard } from 'react-chessboard';
import { Chess } from 'chess.js';
import AnalysisMoves from './AnalysisMoves';
import Bar from '../components/Bar';
import './Analyzed.css';

const options = [
  { value: 'game', label: 'Played move'}, 
  { value: '500', label: '500 ELO most human move'},
  { value: '1000', label: '1000 ELO most human move'}, 
  { value: '1500', label: '1500 ELO most human move'}
]

const Analyzed = () => {
  const [fen, setFen] = useState('start');
  const chess = useRef(new Chess());
  const [index, setIndex] = useState(0);
  // const moveIndex = useRef(0);
  const [moves, setMoves] = useState([]);
  const location = useLocation();
  const pgn = location.state.pgn;
  const moveIndexToFeedback = useRef({});
  const [selected, setSelected] = useState({ value: 'game', label: 'Played move'});
  const [arrows, setArrows] = useState([]);
  const [feedback, setFeedback] = useState("");

  const getPgnMoves = (pgn) => {
    const newChess = new Chess();
    newChess.loadPgn(pgn);
    console.log(newChess.history());
    return newChess.history();
  };

  const handleChange = (option) => {
    setSelected(option);
  }

  useEffect(() => {
    if (selected.value === "game" && moves.length > 0) {
      console.log("yes");
      const move_info = chess.current.move(moves[index]);
      setArrows([[move_info["from"], move_info["to"]]])
      chess.current.undo();
    }
  }, [selected, index, moves])

  const generateExplanation = () => {
    setFeedback("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
  }

  const updateMove = (x) => {
    chess.current.reset();
    for (let i = 0; i < x; i++) {
      chess.current.move(moves[i]);
    }
    setFen(chess.current.fen());
    setIndex(x);
  }

  const makeMove = () => {
    if (index >= moves.length) {
      return;
    }

    chess.current.move(moves[index]);
    setFen(chess.current.fen());
    setIndex(index + 1);

    // // if not cached, request for feedback
    // if (!moveIndexToFeedback.current[index]) {
    //   const model_input = [];
    //   const history = chess.history({ verbose: true }).slice(0, index);
    //   const moves_made = moves.slice(0, index); // gets all moves so far
    //   for (let i = 0; i < index; i++) {
    //     model_input.append((history[i].from, moves_made.slice(Math.max(0, i - 16), i), history[i].color));
    //   }

    //   // get feedback from model
    //   // const feedback = model(model_input);
    // } else {
    //   // display cached feedback
    //   setFeedback(moveIndexToFeedback.current[index.current]);
    // }

    // store feedback in hashmap
    moveIndexToFeedback.current[index] = feedback;
  }

  const undoMove = () => {
    if (index <= 0) {
      return;
    }
    chess.current.undo();
    setFen(chess.current.fen());
    setIndex(index - 1);

    // get cached feedback
    // setFeedback(moveIndexToFeedback.current[index]);
    
    // display feedback
  }

  useEffect(() => {
    const handleKeydown = (event) => {
      if (event.key === 'ArrowRight' && index < moves.length) {
        makeMove();
      } else if (event.key === 'ArrowLeft' && index > 0) {
        undoMove();
      }
    };

    window.addEventListener('keydown', handleKeydown);
    return () => window.removeEventListener('keydown', handleKeydown);
  }, [moves, index])

  useEffect(() => {
    setMoves(getPgnMoves(pgn)); // format: ['e4', 'e5', ..., 'Nf3', 'Nc6']
  }, [pgn]);

  return (
    <div className="analysis-page-layout">
      <div className="analysis-header">
        <h2>Analysis</h2>
      </div>
      <div className="analysis-page-container">
        {/* <Progress.Line showInfo={false} strokeColor={"white"} vertical={true}/> */}
        <Bar color={"black"} value={50} label={"+0.0"}/>
        <Bar color={"blue"} value={80} label={"hard"}/>
        <div className="chessboard-container">
          <Chessboard position={fen} areArrowsAllowed={false} arePiecesDraggable={false} boardWidth={560} customArrows={arrows}/>
        </div>
        <div className="analysis-right-container">
          <AnalysisMoves moves={moves} index={index} updateMove={updateMove}/>
          <div className="analysis-button-container">
            <button className="analysis-back"><img src="/left.png" onClick={undoMove} alt="left" height="50"/></button>
            <button className="analysis-move"><img src="/right.png" onClick={makeMove} alt="right" height="50"/></button>
          </div>
        </div>
        <div className="analysis-explanation-container">
          <div className="explanation-selector-container">
            <Select className="analysis-select" value={selected} onChange={handleChange} options={options} />
            { selected.value !== "game" ? 
              <p> Most likely move: filler </p>
            : <p> Next played move: {moves[index]} </p>
            }
            <button onClick={(e) => generateExplanation()}>Explain</button>
          </div>
          {feedback.length > 0 ? 
          <div className="explanation-container">
            <span>{feedback}</span>
          </div> 
          : <Fragment />}
        </div>
      </div>
    </div>
  );
};  
export default Analyzed;