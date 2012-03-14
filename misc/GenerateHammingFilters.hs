{-# LANGUAGE OverloadedStrings #-}
import Control.Arrow ((&&&), first, second)
import Data.List (foldl')
import Data.Int (Int64)
import Data.Monoid
import qualified Data.Text.Lazy as T
import qualified Data.Text.Lazy.IO as TIO
import qualified Data.Text.Lazy.Builder as B

bitsToInt :: [Bool] -> Int
bitsToInt = foldl' ((+) . (*2)) 0 . map fromEnum

allNBitNumbers :: Int -> Int -> [[Bool]]
allNBitNumbers _ 0 = [[]]
allNBitNumbers 0 numBits = [replicate numBits False]
allNBitNumbers numOnes numBits
  | numOnes > numBits = [[]]
  | numOnes == numBits = with1
  | otherwise = with1 ++ with0
  where with0 = map (False:) (allNBitNumbers numOnes (numBits - 1))
        with1 = map (True:) (allNBitNumbers (numOnes-1) (numBits - 1))

upToNOnes :: Int -> Int -> [(Int, [[Bool]])]
upToNOnes numOnes numBits = map aux [1..numOnes]
  where aux = (length&&&id) . flip allNBitNumbers numBits

intercalateBuild :: T.Text -> [T.Text] -> B.Builder
intercalateBuild sep = go
  where go [] = mempty
        go (x:[]) = B.fromLazyText x
        go (x:xs) = B.fromLazyText x <> sepB <> go xs
        sepB = B.fromLazyText sep

generateFilters :: Int -> B.Builder
generateFilters n = count
  where count = mconcat [ B.fromLazyText "uint32_t filterCounts"
                        , B.fromString (show n)
                        , B.fromLazyText "[] = {"
                        , intercalateBuild ", " allNeighborCounts
                        , B.fromLazyText "};\nuint32_t filter"
                        , B.fromString (show n)
                        , B.fromLazyText "[] = {"
                        , intercalateBuild ", " allNeighborMasks
                        , B.fromLazyText "};\n\n" ]
        masks = upToNOnes (min 4 n) n
        showBits = T.pack . show . bitsToInt
        allNeighborMasks = concatMap (map showBits . snd) masks
        allNeighborCounts = map (T.pack . show . fst) masks

lineLengthConvention :: Int64 -> T.Text -> T.Text
lineLengthConvention len = mconcat . map breakLine . T.lines
  where breakLine t
          | T.length t < 80 = t <> "\n"
          | otherwise = chunkTokens 0 (nextToken t)
        chunkTokens :: Int64 -> (T.Text, T.Text) -> T.Text
        chunkTokens n (nxt,rst)
          | n + nxtLen > len = 
            "\n" <> nxt <> if T.null rst 
                           then "\n" 
                           else ", " <> chunkTokens nxtLen (nextToken rst)
          | T.null rst = nxt <> "\n"
          | otherwise = nxt <> ", " <> chunkTokens (n+nxtLen) (nextToken rst)
          where nxtLen = T.length nxt + 2
        nextToken = second (T.drop 2) . T.breakOn ", "

main = TIO.writeFile "HammingNeighborhoodFilters.h" . lineLengthConvention 80 $
       "#ifndef HAMMINGNEIGHBORHOODFILTERS_H\n" <>
       "#define HAMMINGNEIGHBORHOODFILTERS_H\n\n" <>
       docs <>
       B.toLazyText (mconcat (map generateFilters [3..12])) <>
       "#endif"

docs :: T.Text
docs = T.intercalate "\n"
       [ "/*" 
       , " * These filters store precomputed hamming-adjacency information. The"
       , " * filterCountsN arrays store the number of K-neighbors for N-bit"
       , " * codes. For example, filterCounts3 says that 3-bit codes have 3"
       , " * neighbors with distance 1, 3 neighbors with distance 2, and 1"
       , " * neighbor with distance 3. The filterN arrays then encode the"
       , " * adjacency information in terms of bitmasks to compute neighbors of"
       , " * a given code. For instance, to compute the 1-neighbors of the 3-bit"
       , " * code 010, we XOR the code with 4 (100) to yield 110, then with 2"
       , " * (010) to yield 000, and finally with 1 (001) to yield 011. These"
       , " * are the 3 1-bit distanct neighbors of 010."
       , " *"
       , " * The arrays are computed by considering all the one-bit numbers of"
       , " * the given bit-length, then all the two-bit numbers, and so on. For"
       , " * instance, an 8-bit representation of 15 is 00001111, which may be"
       , " * used to compute a 4-neighbor of a code via an XOR operation"
       , " * (e.g. 11000010 XOR 00001111 = 11001101 which has 4 bits flipped"
       , " * from the code 11000010)."
       , " */\n\n" ]
