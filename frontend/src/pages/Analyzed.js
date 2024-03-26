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
    setArrows([]);
    setFeedback("");
  }

  useEffect(() => {
    if (index >= moves.length) {
      setArrows([]);
    }
    else if (selected.value === "game" && moves.length > 0) {
      const move_info = chess.current.move(moves[index]);
      setArrows([[move_info["from"], move_info["to"]]])
      chess.current.undo();
    }
  }, [selected, index, moves])

  const generateExplanation = () => {
    setFeedback("The castle move by Black (O-O-O), though it appears to result in some slight short term losses, might set up a more advantageous strategic situation in the long run. The material count remains unchanged at 5559, indicating no pieces were lost or exchanged in this move. The move increases Black's control over the board from 37 to 46 squares, implying that Black has expanded their influence on the game. While the castle move moved the king further away from the nearest black pawn, thus potentially leading to a slightly more exposed position, the shift in control indicates a broader control of the board. However, the move leaves one of Black's pieces hanging, potentially creating a risk for the next moves. This move might signal a transition from a defensive to more offensive play for Black.")
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
        <Bar color={"black"} value={25} label={"easy"}/>
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
              <p>{ chess.current.isCheckmate() ? "Checkmate!" : "Most likely move: filler" } </p> :
              <p>{ index < moves.length ? "Next played move: " + moves[index] : "Game is over!" }</p>
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