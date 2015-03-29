/***********************************************************************
Moses - factored phrase-based language decoder
Copyright (C) 2014- University of Edinburgh

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
***********************************************************************/

#include <cmath>
#include <limits>
#include <list>

#include <boost/unordered_set.hpp>

#include "util/file_piece.hh"
#include "util/tokenize_piece.hh"

#include "BleuScorer.h"
#include "ForestRescore.h"

using namespace std;

namespace MosesTuning
{

std::ostream& operator<<(std::ostream& out, const WordVec& wordVec)
{
  out << "[";
  for (size_t i = 0; i < wordVec.size(); ++i) {
    out << wordVec[i]->first;
    if (i+1< wordVec.size()) out << " ";
  }
  out << "]";
  return out;
}


void ReferenceSet::Load(const vector<string>& files, Vocab& vocab)
{
  for (size_t i = 0; i < files.size(); ++i) {
    util::FilePiece fh(files[i].c_str());
    size_t sentenceId = 0;
    while(true) {
      StringPiece line;
      try {
        line = fh.ReadLine();
      } catch (util::EndOfFileException &e) {
        break;
      }
      AddLine(sentenceId, line, vocab);
      ++sentenceId;
    }
  }

}

void ReferenceSet::AddLine(size_t sentenceId, const StringPiece& line, Vocab& vocab)
{
  //cerr << line << endl;
  NgramCounter ngramCounts;
  list<WordVec> openNgrams;
  size_t length = 0;
  //tokenize & count
  for (util::TokenIter<util::SingleCharacter, true> j(line, util::SingleCharacter(' ')); j; ++j) {
    const Vocab::Entry* nextTok = &(vocab.FindOrAdd(*j));
    ++length;
    openNgrams.push_front(WordVec());
    for (list<WordVec>::iterator k = openNgrams.begin(); k != openNgrams.end();  ++k) {
      k->push_back(nextTok);
      ++ngramCounts[*k];
    }
    if (openNgrams.size() >=  kBleuNgramOrder) openNgrams.pop_back();
  }

  //merge into overall ngram map
  for (NgramCounter::const_iterator ni = ngramCounts.begin();
       ni != ngramCounts.end(); ++ni) {
    size_t count = ni->second;
    //cerr << *ni << " " << count <<  endl;
    if (ngramCounts_.size() <= sentenceId) ngramCounts_.resize(sentenceId+1);
    NgramMap::iterator totalsIter = ngramCounts_[sentenceId].find(ni->first);
    if (totalsIter == ngramCounts_[sentenceId].end()) {
      ngramCounts_[sentenceId][ni->first] = pair<size_t,size_t>(count,count);
    } else {
      ngramCounts_[sentenceId][ni->first].first = max(count, ngramCounts_[sentenceId][ni->first].first); //clip
      ngramCounts_[sentenceId][ni->first].second += count; //no clip
    }
  }
  //length
  if (lengths_.size() <= sentenceId) lengths_.resize(sentenceId+1);
  //TODO - length strategy - this is MIN
  if (!lengths_[sentenceId]) {
    lengths_[sentenceId] = length;
  } else {
    lengths_[sentenceId] = min(length,lengths_[sentenceId]);
  }
  //cerr << endl;

}

size_t ReferenceSet::NgramMatches(size_t sentenceId, const WordVec& ngram, bool clip) const
{
  const NgramMap& ngramCounts = ngramCounts_.at(sentenceId);
  NgramMap::const_iterator ngi = ngramCounts.find(ngram);
  if (ngi == ngramCounts.end()) return 0;
  return clip ? ngi->second.first : ngi->second.second;
}

VertexState::VertexState(): bleuStats(kBleuNgramOrder), targetLength(0) {}

void HgBleuScorer::UpdateMatches(const NgramCounter& counts, vector<FeatureStatsType>& bleuStats ) const
{
  for (NgramCounter::const_iterator ngi = counts.begin(); ngi != counts.end(); ++ngi) {
    //cerr << "Checking: " << *ngi << " matches " << references_.NgramMatches(sentenceId_,*ngi,false) <<  endl;
    size_t order = ngi->first.size();
    size_t count = ngi->second;
    bleuStats[(order-1)*2 + 1] += count;
    bleuStats[(order-1) * 2] += min(count, references_.NgramMatches(sentenceId_,ngi->first,false));
  }
}

size_t HgBleuScorer::GetTargetLength(const Edge& edge) const
{
  size_t targetLength = 0;
  for (size_t i = 0; i < edge.Words().size(); ++i) {
    const Vocab::Entry* word = edge.Words()[i];
    if (word) ++targetLength;
  }
  for (size_t i = 0; i < edge.Children().size(); ++i) {
    const VertexState& state = vertexStates_[edge.Children()[i]];
    targetLength += state.targetLength;
  }
  return targetLength;
}

FeatureStatsType HgBleuScorer::Score(const Edge& edge, const Vertex& head, vector<FeatureStatsType>& bleuStats)
{
  NgramCounter ngramCounts;
  size_t childId = 0;
  size_t wordId = 0;
  size_t contextId = 0; //position within left or right context
  const VertexState* vertexState = NULL;
  bool inLeftContext = false;
  bool inRightContext = false;
  list<WordVec> openNgrams;
  const Vocab::Entry* currentWord = NULL;
  while (wordId < edge.Words().size()) {
    currentWord = edge.Words()[wordId];
    if (currentWord != NULL) {
      ++wordId;
    } else {
      if (!inLeftContext && !inRightContext) {
        //entering a vertex
        assert(!vertexState);
        vertexState = &(vertexStates_[edge.Children()[childId]]);
        ++childId;
        if (vertexState->leftContext.size()) {
          inLeftContext = true;
          contextId = 0;
          currentWord = vertexState->leftContext[contextId];
        } else {
          //empty context
          vertexState = NULL;
          ++wordId;
          continue;
        }
      } else {
        //already in a vertex
        ++contextId;
        if (inLeftContext && contextId < vertexState->leftContext.size()) {
          //still in left context
          currentWord = vertexState->leftContext[contextId];
        } else if (inLeftContext) {
          //at end of left context
          if (vertexState->leftContext.size() == kBleuNgramOrder-1) {
            //full size context, jump to right state
            openNgrams.clear();
            inLeftContext = false;
            inRightContext = true;
            contextId = 0;
            currentWord = vertexState->rightContext[contextId];
          } else {
            //short context, just ignore right context
            inLeftContext = false;
            vertexState = NULL;
            ++wordId;
            continue;
          }
        } else {
          //in right context
          if (contextId < vertexState->rightContext.size()) {
            currentWord = vertexState->rightContext[contextId];
          } else {
            //leaving vertex
            inRightContext = false;
            vertexState = NULL;
            ++wordId;
            continue;
          }
        }
      }
    }
    assert(currentWord);
    if (graph_.IsBoundary(currentWord)) continue;
    openNgrams.push_front(WordVec());
    openNgrams.front().reserve(kBleuNgramOrder);
    for (list<WordVec>::iterator k = openNgrams.begin(); k != openNgrams.end();  ++k) {
      k->push_back(currentWord);
      //Only insert ngrams that cross boundaries
      if (!vertexState || (inLeftContext && k->size() > contextId+1)) ++ngramCounts[*k];
    }
    if (openNgrams.size() >=  kBleuNgramOrder) openNgrams.pop_back();
  }

  //Collect matches
  //This edge
  //cerr << "edge ngrams" << endl;
  UpdateMatches(ngramCounts, bleuStats);

  //Child vertexes
  for (size_t i = 0; i < edge.Children().size(); ++i) {
    //cerr << "vertex ngrams " << edge.Children()[i] << endl;
    for (size_t j = 0; j < bleuStats.size(); ++j) {
      bleuStats[j] += vertexStates_[edge.Children()[i]].bleuStats[j];
    }
  }


  FeatureStatsType sourceLength = head.SourceCovered();
  size_t referenceLength = references_.Length(sentenceId_);
  FeatureStatsType effectiveReferenceLength =
    sourceLength / totalSourceLength_ * referenceLength;

  bleuStats[bleuStats.size()-1] = effectiveReferenceLength;
  //backgroundBleu_[backgroundBleu_.size()-1] =
  //  backgroundRefLength_ * sourceLength / totalSourceLength_;
  FeatureStatsType bleu = sentenceLevelBackgroundBleu(bleuStats, backgroundBleu_);

  return bleu;
}

void HgBleuScorer::UpdateState(const Edge& winnerEdge, size_t vertexId, const vector<FeatureStatsType>& bleuStats)
{
  //TODO: Maybe more efficient to absorb into the Score() method
  VertexState& vertexState = vertexStates_[vertexId];
  //cerr << "Updating state for " << vertexId << endl;

  //leftContext
  int wi = 0;
  const VertexState* childState = NULL;
  int contexti = 0; //index within child context
  int childi = 0;
  while (vertexState.leftContext.size() < (kBleuNgramOrder-1)) {
    if ((size_t)wi >= winnerEdge.Words().size()) break;
    const Vocab::Entry* word = winnerEdge.Words()[wi];
    if (word != NULL) {
      vertexState.leftContext.push_back(word);
      ++wi;
    } else {
      if (childState == NULL) {
        //start of child state
        childState = &(vertexStates_[winnerEdge.Children()[childi++]]);
        contexti = 0;
      }
      if ((size_t)contexti < childState->leftContext.size()) {
        vertexState.leftContext.push_back(childState->leftContext[contexti++]);
      } else {
        //end of child context
        childState = NULL;
        ++wi;
      }
    }
  }

  //rightContext
  wi = winnerEdge.Words().size() - 1;
  childState = NULL;
  childi = winnerEdge.Children().size() - 1;
  while (vertexState.rightContext.size() < (kBleuNgramOrder-1)) {
    if (wi < 0) break;
    const Vocab::Entry* word = winnerEdge.Words()[wi];
    if (word != NULL) {
      vertexState.rightContext.push_back(word);
      --wi;
    } else {
      if (childState == NULL) {
        //start (ie rhs) of child state
        childState = &(vertexStates_[winnerEdge.Children()[childi--]]);
        contexti = childState->rightContext.size()-1;
      }
      if (contexti >= 0) {
        vertexState.rightContext.push_back(childState->rightContext[contexti--]);
      } else {
        //end (ie lhs) of child context
        childState = NULL;
        --wi;
      }
    }
  }
  reverse(vertexState.rightContext.begin(), vertexState.rightContext.end());

  //length + counts
  vertexState.targetLength = GetTargetLength(winnerEdge);
  vertexState.bleuStats = bleuStats;
}


typedef pair<const Edge*,FeatureStatsType> BackPointer;


/**
 * Recurse through back pointers
 **/
static void GetBestHypothesis(size_t vertexId, const Graph& graph, const vector<BackPointer>& bps,
                              HgHypothesis* bestHypo)
{
  //cerr << "Expanding " << vertexId << " Score: " << bps[vertexId].second << endl;
  //UTIL_THROW_IF(bps[vertexId].second == kMinScore+1, HypergraphException, "Landed at vertex " << vertexId << " which is a dead end");
  if (!bps[vertexId].first) return;
  const Edge* prevEdge = bps[vertexId].first;
  bestHypo->featureVector += *(prevEdge->Features().get());
  size_t childId = 0;
  for (size_t i = 0; i < prevEdge->Words().size(); ++i) {
    if (prevEdge->Words()[i] != NULL) {
      bestHypo->text.push_back(prevEdge->Words()[i]);
    } else {
      size_t childVertexId = prevEdge->Children()[childId++];
      HgHypothesis childHypo;
      GetBestHypothesis(childVertexId,graph,bps,&childHypo);
      bestHypo->text.insert(bestHypo->text.end(), childHypo.text.begin(), childHypo.text.end());
      bestHypo->featureVector += childHypo.featureVector;
    }
  }
}

void Viterbi(const Graph& graph, const SparseVector& weights, float bleuWeight, const ReferenceSet& references , size_t sentenceId, const std::vector<FeatureStatsType>& backgroundBleu,  HgHypothesis* bestHypo)
{
  BackPointer init(NULL,kMinScore);
  vector<BackPointer> backPointers(graph.VertexSize(),init);
  HgBleuScorer bleuScorer(references, graph, sentenceId, backgroundBleu);
  vector<FeatureStatsType> winnerStats(kBleuNgramOrder*2+1);
  for (size_t vi = 0; vi < graph.VertexSize(); ++vi) {
//    cerr << "vertex id " << vi <<  endl;
    FeatureStatsType winnerScore = kMinScore;
    const Vertex& vertex = graph.GetVertex(vi);
    const vector<const Edge*>& incoming = vertex.GetIncoming();
    if (!incoming.size()) {
      //UTIL_THROW(HypergraphException, "Vertex " << vi << " has no incoming edges");
      //If no incoming edges, vertex is a dead end
      backPointers[vi].first = NULL;
      backPointers[vi].second = kMinScore;
    } else {
      //cerr << "\nVertex: " << vi << endl;
      for (size_t ei = 0; ei < incoming.size(); ++ei) {
        //cerr << "edge id " << ei << endl;
        FeatureStatsType incomingScore = incoming[ei]->GetScore(weights);
        for (size_t i = 0; i < incoming[ei]->Children().size(); ++i) {
          size_t childId = incoming[ei]->Children()[i];
          //UTIL_THROW_IF(backPointers[childId].second == kMinScore,
          //  HypergraphException, "Graph was not topologically sorted. curr=" << vi << " prev=" << childId);
          incomingScore = max(incomingScore + backPointers[childId].second, kMinScore);
        }
        vector<FeatureStatsType> bleuStats(kBleuNgramOrder*2+1);
        // cerr << "Score: " << incomingScore << " Bleu: ";
        // if (incomingScore > nonbleuscore) {nonbleuscore = incomingScore; nonbleuid = ei;}
        FeatureStatsType totalScore = incomingScore;
        if (bleuWeight) {
          FeatureStatsType bleuScore = bleuScorer.Score(*(incoming[ei]), vertex, bleuStats);
          if (isnan(bleuScore)) {
            cerr << "WARN: bleu score undefined" << endl;
            cerr << "\tVertex id : " << vi << endl;
            cerr << "\tBleu stats : ";
            for (size_t i = 0; i < bleuStats.size(); ++i) {
              cerr << bleuStats[i] << ",";
            }
            cerr << endl;
            bleuScore = 0;
          }
          //UTIL_THROW_IF(isnan(bleuScore), util::Exception, "Bleu score undefined, smoothing problem?");
          totalScore += bleuWeight * bleuScore;
          //  cerr << bleuScore << " Total: " << incomingScore << endl << endl;
          //cerr << "is " << incomingScore << " bs " << bleuScore << endl;
        }
        if (totalScore >= winnerScore) {
          //We only store the feature score (not the bleu score) with the vertex,
          //since the bleu score is always cumulative, ie from counts for the whole span.
          winnerScore = totalScore;
          backPointers[vi].first = incoming[ei];
          backPointers[vi].second = incomingScore;
          winnerStats = bleuStats;
        }
      }
      //update with winner
      //if (bleuWeight) {
      //TODO: Not sure if we need this when computing max-model solution
      if (backPointers[vi].first) {
        bleuScorer.UpdateState(*(backPointers[vi].first), vi, winnerStats);
      }

    }
//    cerr  << "backpointer[" << vi << "] = (" << backPointers[vi].first << "," << backPointers[vi].second << ")" << endl;
  }

  //expand back pointers
  GetBestHypothesis(graph.VertexSize()-1, graph, backPointers, bestHypo);

  //bleu stats and fv

  //Need the actual (clipped) stats
  //TODO: This repeats code in bleu scorer - factor out
  bestHypo->bleuStats.resize(kBleuNgramOrder*2+1);
  NgramCounter counts;
  list<WordVec> openNgrams;
  for (size_t i = 0; i < bestHypo->text.size(); ++i) {
    const Vocab::Entry* entry = bestHypo->text[i];
    if (graph.IsBoundary(entry)) continue;
    openNgrams.push_front(WordVec());
    for (list<WordVec>::iterator k = openNgrams.begin(); k != openNgrams.end();  ++k) {
      k->push_back(entry);
      ++counts[*k];
    }
    if (openNgrams.size() >=  kBleuNgramOrder) openNgrams.pop_back();
  }
  for (NgramCounter::const_iterator ngi = counts.begin(); ngi != counts.end(); ++ngi) {
    size_t order = ngi->first.size();
    size_t count = ngi->second;
    bestHypo->bleuStats[(order-1)*2 + 1] += count;
    bestHypo->bleuStats[(order-1) * 2] += min(count, references.NgramMatches(sentenceId,ngi->first,true));
  }
  bestHypo->bleuStats[kBleuNgramOrder*2] = references.Length(sentenceId);
}

typedef pair<const Edge*,FeatureStatsType> ForwardPointer;

static void ExtendBestHypothesis(size_t vertexId, const Graph& graph, const vector<BackPointer>& bps, const vector<ForwardPointer>& fps,
    const map<const Edge*, size_t>& edgeHeads, HgHypothesis* bestHypo)
{
  //cerr << "Expanding " << vertexId << " Score: " << bps[vertexId].second << endl;
  //UTIL_THROW_IF(bps[vertexId].second == kMinScore+1, HypergraphException, "Landed at vertex " << vertexId << " which is a dead end");
  if (!fps[vertexId].first) return;
  const Edge* nextEdge = fps[vertexId].first;
  //bestHypo->featureVector += *(nextEdge->Features().get());
  size_t childId = 0;
  bool left = true;
  int count = 0;
  for (size_t i = 0; i < nextEdge->Words().size(); ++i) {
    if (nextEdge->Words()[i] != NULL) {
      if (!left) {
        bestHypo->text.push_back(nextEdge->Words()[i]);
      } else {
        bestHypo->text.insert(bestHypo->text.begin()+count, nextEdge->Words()[i]);
        count++;
      }
    } else {
      size_t childVertexId = nextEdge->Children()[childId++];
      if (childVertexId == vertexId) {
        left = false;
      } else {
        HgHypothesis childHypo;
        GetBestHypothesis(childVertexId,graph,bps,&childHypo);
        if (!left) {
          bestHypo->text.insert(bestHypo->text.end(), childHypo.text.begin(), childHypo.text.end());
        } else {
          bestHypo->text.insert(bestHypo->text.begin()+count, childHypo.text.begin(), childHypo.text.end());
          count += childHypo.text.size();
        }
        //bestHypo->featureVector += childHypo.featureVector;
      }
    }
  }
  ExtendBestHypothesis(edgeHeads.find(nextEdge)->second, graph, bps, fps, edgeHeads, bestHypo);
}


void ViterbiForSA(const Graph& graph, const SparseVector& weights, float bleuWeight, const ReferenceSet& references , size_t sentenceId, const std::vector<FeatureStatsType>& backgroundBleu,  HypColl& bestHypos)
{
  size_t size = graph.GetVertex(graph.VertexSize()-1).SourceCovered();
  //bestHypos.resize(size);
  //for(size_t i = 0; i < size; i++) {
  //  bestHypos[i].resize(size-i, HgHypothesis());
  //}

  //std::set<pair<size_t, size_t> > exists;

    map<const Edge*, size_t> edgeHeads;
    vector<FeatureStatsType> vertexBackwardScores(graph.VertexSize(), kMinScore);
    vector<vector<const Edge*> > outgoing(graph.VertexSize());

    //Compute backward scores
    for (size_t vi = 0; vi < graph.VertexSize(); ++vi) {
      // cerr << "Vertex " << vi << endl;
      const Vertex& vertex = graph.GetVertex(vi);
      const vector<const Edge*>& incoming = vertex.GetIncoming();
      if (!incoming.size()) {
        vertexBackwardScores[vi] = 0;
      } else {
        for (size_t ei = 0; ei < incoming.size(); ++ei) {
          //cerr << "Edge " << edgeIds[incoming[ei]] << endl;
          edgeHeads[incoming[ei]]= vi;
          FeatureStatsType incomingScore = incoming[ei]->GetScore(weights);
          for (size_t i = 0; i < incoming[ei]->Children().size(); ++i) {
            //cerr << "\tChild " << incoming[ei]->Children()[i] << endl;
            size_t childId = incoming[ei]->Children()[i];
            UTIL_THROW_IF(vertexBackwardScores[childId] == kMinScore,
                          HypergraphException, "Graph was not topologically sorted. curr=" << vi << " prev=" << childId);
            outgoing[childId].push_back(incoming[ei]);
            incomingScore += vertexBackwardScores[childId];
          }
          //cerr << "Backward score: " << incomingScore << endl;
          if (incomingScore > vertexBackwardScores[vi]) vertexBackwardScores[vi] = incomingScore;
        }
      }
    }

    //Compute forward scores
    ForwardPointer initfp(NULL,kMinScore);
    vector<ForwardPointer> forwardPointers(graph.VertexSize(),initfp);

    for (size_t i = 1; i <= graph.VertexSize(); ++i) {
      size_t vi = graph.VertexSize() - i;
      //cerr << "Vertex " << vi << endl;
      if (!outgoing[vi].size()) {
        forwardPointers[vi].first = NULL;
        forwardPointers[vi].second = 0;
      } else {
        for (size_t ei = 0; ei < outgoing[vi].size(); ++ei) {
          //cerr << "Edge " << edgeIds[outgoing[vi][ei]] << endl;
          FeatureStatsType outgoingScore = 0;
          //add score of head
          outgoingScore += forwardPointers[edgeHeads[outgoing[vi][ei]]].second;
          //sum scores of siblings
          for (size_t k = 0; k < outgoing[vi][ei]->Children().size(); ++k) {
            size_t siblingId = outgoing[vi][ei]->Children()[k];
            if (siblingId != vi) {
              //cerr << "\tSibling " << siblingId << endl;
              outgoingScore += vertexBackwardScores[siblingId];
            }
          }
          outgoingScore += outgoing[vi][ei]->GetScore(weights);
          if (outgoingScore > forwardPointers[vi].second) {
            forwardPointers[vi].first = outgoing[vi][ei];
            forwardPointers[vi].second = outgoingScore;
          }
          //cerr << "Vertex " << vi << " forward score " << outgoingScore << endl;
        }
      }
    }

  map<Range, size_t> rangeVi;

  BackPointer init(NULL,kMinScore);
  vector<BackPointer> backPointers(graph.VertexSize(),init);
  HgBleuScorer bleuScorer(references, graph, sentenceId, backgroundBleu);
  vector<FeatureStatsType> winnerStats(kBleuNgramOrder*2+1);
  for (size_t vi = 0; vi < graph.VertexSize(); ++vi) {
//    cerr << "vertex id " << vi <<  endl;
    FeatureStatsType winnerScore = kMinScore;
    const Vertex& vertex = graph.GetVertex(vi);
    const vector<const Edge*>& incoming = vertex.GetIncoming();
    if (!incoming.size()) {
      //UTIL_THROW(HypergraphException, "Vertex " << vi << " has no incoming edges");
      //If no incoming edges, vertex is a dead end
      backPointers[vi].first = NULL;
      backPointers[vi].second = kMinScore;
    } else {
      //cerr << "\nVertex: " << vi << endl;
      for (size_t ei = 0; ei < incoming.size(); ++ei) {

        edgeHeads[incoming[ei]] = vi;

        //cerr << "edge id " << ei << endl;
        FeatureStatsType incomingScore = incoming[ei]->GetScore(weights);
        for (size_t i = 0; i < incoming[ei]->Children().size(); ++i) {
          size_t childId = incoming[ei]->Children()[i];
          //UTIL_THROW_IF(backPointers[childId].second == kMinScore,
          //  HypergraphException, "Graph was not topologically sorted. curr=" << vi << " prev=" << childId);
          incomingScore = max(incomingScore + backPointers[childId].second, kMinScore);
          outgoing[childId].push_back(incoming[ei]);
        }
        vector<FeatureStatsType> bleuStats(kBleuNgramOrder*2+1);
        // cerr << "Score: " << incomingScore << " Bleu: ";
        // if (incomingScore > nonbleuscore) {nonbleuscore = incomingScore; nonbleuid = ei;}
        FeatureStatsType totalScore = incomingScore;
        if (bleuWeight) {
          // #########################
          /*const Edge* temp = backPointers[vi].first;
          backPointers[vi].first = incoming[ei];
          boost::shared_ptr<HgHypothesis> bestHypo (new HgHypothesis());
          bool flag = GetBestHypothesis(vi, graph, backPointers, bestHypo.get());
          ExtendBestHypothesis(vi, graph, backPointers, forwardPointers, edgeHeads, bestHypo.get());
          // update BLEU
          (*bestHypo).bleuStats.resize(kBleuNgramOrder*2+1);
          NgramCounter counts;
          list<WordVec> openNgrams;
          for (size_t i = 0; i < (*bestHypo).text.size(); ++i) {
            const Vocab::Entry* entry = (*bestHypo).text[i];
            if (graph.IsBoundary(entry)) continue;
            openNgrams.push_front(WordVec());
            for (list<WordVec>::iterator k = openNgrams.begin(); k != openNgrams.end();  ++k) {
              k->push_back(entry);
              ++counts[*k];
            }
            if (openNgrams.size() >=  kBleuNgramOrder) openNgrams.pop_back();
          }
          for (NgramCounter::const_iterator ngi = counts.begin(); ngi != counts.end(); ++ngi) {
            size_t order = ngi->first.size();
            size_t count = ngi->second;
            (*bestHypo).bleuStats[(order-1)*2 + 1] += count;
            (*bestHypo).bleuStats[(order-1) * 2] += min(count, references.NgramMatches(sentenceId,ngi->first,true));
          }
          (*bestHypo).bleuStats[kBleuNgramOrder*2] = references.Length(sentenceId);
          backPointers[vi].first = temp;
          FeatureStatsType bleuScore = sentenceLevelBackgroundBleu((*bestHypo).bleuStats, backgroundBleu);
          */// ###############################

          FeatureStatsType bleuScore =
          bleuScorer.Score(*(incoming[ei]), vertex, bleuStats);
          if (isnan(bleuScore)) {
            cerr << "WARN: bleu score undefined" << endl;
            cerr << "\tVertex id : " << vi << endl;
            cerr << "\tBleu stats : ";
            for (size_t i = 0; i < bleuStats.size(); ++i) {
              cerr << bleuStats[i] << ",";
            }
            cerr << endl;
            bleuScore = 0;
          }
          //UTIL_THROW_IF(isnan(bleuScore), util::Exception, "Bleu score undefined, smoothing problem?");
          totalScore += bleuWeight * bleuScore;
          //  cerr << bleuScore << " Total: " << incomingScore << endl << endl;
          //cerr << "is " << incomingScore << " bs " << bleuScore << endl;
        }
        if (totalScore >= winnerScore) {
          //We only store the feature score (not the bleu score) with the vertex,
          //since the bleu score is always cumulative, ie from counts for the whole span.
          winnerScore = totalScore;
          backPointers[vi].first = incoming[ei];
          backPointers[vi].second = incomingScore;
          winnerStats = bleuStats;

          boost::shared_ptr<HgHypothesis> bestHypo (new HgHypothesis());
          GetBestHypothesis(vi, graph, backPointers, bestHypo.get());

          /*(*bestHypo).bleuStats.resize(kBleuNgramOrder*2+1);
          NgramCounter counts;
          list<WordVec> openNgrams;
          for (size_t i = 0; i < (*bestHypo).text.size(); ++i) {
            const Vocab::Entry* entry = (*bestHypo).text[i];
            if (graph.IsBoundary(entry)) continue;
            openNgrams.push_front(WordVec());
            for (list<WordVec>::iterator k = openNgrams.begin(); k != openNgrams.end();  ++k) {
              k->push_back(entry);
              ++counts[*k];
            }
            if (openNgrams.size() >=  kBleuNgramOrder) openNgrams.pop_back();
          }
          for (NgramCounter::const_iterator ngi = counts.begin(); ngi != counts.end(); ++ngi) {
            size_t order = ngi->first.size();
            size_t count = ngi->second;
            (*bestHypo).bleuStats[(order-1)*2 + 1] += count;
            (*bestHypo).bleuStats[(order-1) * 2] += min(count, references.NgramMatches(sentenceId,ngi->first,true));
          }
          (*bestHypo).bleuStats[kBleuNgramOrder*2] = references.Length(sentenceId);
*/
          size_t s = vertex.startPos;
          size_t e = vertex.endPos;

          Range r(s,e);

          //HypColl::iterator iter = bestHypos.find(r);
          //if (iter == bestHypos.end()  || inner_product((*iter->second).featureVector,weights) + bleuWeight*sentenceLevelBackgroundBleu((*iter->second).bleuStats, backgroundBleu) < inner_product((*bestHypo).featureVector,weights) + bleuWeight * sentenceLevelBackgroundBleu((*bestHypo).bleuStats, backgroundBleu)) {
          bestHypos[r] = bestHypo;
          rangeVi[r] = vi;
            //bestHypo.featureVector.write(cerr, " "); cerr << endl;
          //}
        }
      }
      //update with winner
      //if (bleuWeight) {
      //TODO: Not sure if we need this when computing max-model solution
      if (backPointers[vi].first) {
        bleuScorer.UpdateState(*(backPointers[vi].first), vi, winnerStats);
      }

    }
  }

  //HgBleuScorer bleuScorer2(references, graph, sentenceId, backgroundBleu);
/*  for (size_t i = 1; i <= graph.VertexSize(); ++i) {
    size_t vi = graph.VertexSize() - i;
    FeatureStatsType winnerScore = kMinScore;
    const Vertex& vertex = graph.GetVertex(vi);

    if (!outgoing[vi].size()) {
      forwardPointers[vi].first = NULL;
      forwardPointers[vi].second = 0;
    } else {
      for (size_t ei = 0; ei < outgoing[vi].size(); ++ei) {
        //cerr << "Edge " << edgeIds[outgoing[vi][ei]] << endl;
        FeatureStatsType outgoingScore = 0;
        //add score of head
        outgoingScore += forwardPointers[edgeHeads[outgoing[vi][ei]]].second;
        //cerr << "Forward score " << outgoingScore << endl;
        //forwardPointers[edgeHeads[outgoing[vi][ei]]].second = outgoingScore;
        //sum scores of siblings
        for (size_t ci = 0; ci < outgoing[vi][ei]->Children().size(); ++ci) {
          size_t siblingId = outgoing[vi][ei]->Children()[ci];
          if (siblingId != vi) {
            //cerr << "\tSibling " << siblingId << endl;
            outgoingScore += backPointers[siblingId].second;
            //outgoingScore = max(outgoingScore + backPointers[siblingId].second, kMinScore);
          }
        }
        //cerr << outgoingScore << endl;
        outgoingScore += outgoing[vi][ei]->GetScore(weights);
        //outgoing[vi][ei]->Features().get()->write(cerr, " "); cerr << endl;
        //weights.write(cerr, " "); cerr << endl;
        //cerr << outgoing[vi][ei]->GetScore(weights) << endl;
        //cerr << outgoingScore << endl;
        FeatureStatsType totalScore = outgoingScore;

        if (totalScore > winnerScore) {
          forwardPointers[vi].first = outgoing[vi][ei];
          forwardPointers[vi].second = outgoingScore;
          winnerScore = totalScore;
          //winnerStats = bleuStats;
        }
        //cerr << "Vertex " << vi << " forward score " << outgoingScore << endl;
      }
      //if (forwardPointers[vi].first) {
      //  bleuScorer2.UpdateState(*(forwardPointers[vi].first), vi, winnerStats);
      //}
    }
  }
*/


  for(HypColl::iterator iter = bestHypos.begin(); iter != bestHypos.end(); iter++) {
    const Range& range = iter->first;
    HgHypothesis* bestHypo = iter->second.get();

    assert(rangeVi.find(range) != rangeVi.end());

    size_t vi = rangeVi.find(range)->second;

    ExtendBestHypothesis(vi, graph, backPointers, forwardPointers, edgeHeads, bestHypo);

    (*bestHypo).bleuStatsPot.resize(kBleuNgramOrder*2+1);
    NgramCounter counts;
    list<WordVec> openNgrams;
    for (size_t i = 0; i < (*bestHypo).text.size(); ++i) {
      const Vocab::Entry* entry = (*bestHypo).text[i];
      if (graph.IsBoundary(entry)) continue;
      openNgrams.push_front(WordVec());
      for (list<WordVec>::iterator k = openNgrams.begin(); k != openNgrams.end();  ++k) {
        k->push_back(entry);
        ++counts[*k];
      }
      if (openNgrams.size() >=  kBleuNgramOrder) openNgrams.pop_back();
    }
    for (NgramCounter::const_iterator ngi = counts.begin(); ngi != counts.end(); ++ngi) {
      size_t order = ngi->first.size();
      size_t count = ngi->second;
      (*bestHypo).bleuStatsPot[(order-1)*2 + 1] += count;
      (*bestHypo).bleuStatsPot[(order-1) * 2] += min(count, references.NgramMatches(sentenceId,ngi->first,true));
    }
    (*bestHypo).bleuStatsPot[kBleuNgramOrder*2] = references.Length(sentenceId);
  }

  //exit(1);

}

};
