import { PCA } from "ml-pca";

/**
 * Reduce embeddings (1536D) → 2D using PCA
 */
export function reduceTo2D(embeddings) {
  const pca = new PCA(embeddings);
  return pca.predict(embeddings, { nComponents: 2 }).to2DArray();
}