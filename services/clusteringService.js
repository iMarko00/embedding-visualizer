import { DBSCAN } from "density-clustering";
import { kmeans } from "ml-kmeans";

/**
 * Adaptive K computation
 */
function computeAdaptiveK(points, n, opts = {}) {
  const { minK = 2, maxRatio = 0.5, scale = 2 } = opts;
  if (!Array.isArray(points) || points.length === 0) return Math.max(minK, 2);
  const xs = points.map(p => p[0]);
  const ys = points.map(p => p[1]);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);
  const area = Math.max(width * height, 1e-6);
  const density = n / area;
  const kRaw = Math.round(Math.sqrt(n * density) * scale);
  const maxK = Math.max(2, Math.floor(n * maxRatio));
  return Math.min(Math.max(minK, kRaw || minK), Math.max(2, Math.floor(n / 2), maxK));
}

/**
 * Performs DBSCAN with K-means fallback
 */
export function clusterRequirements(reduced, numRequirements) {
  const dbscan = new DBSCAN();
  let eps = 0.2, minPts = 3;
  if (numRequirements > 15) { eps = 0.15; minPts = 5; }

  const dbscanResult = dbscan.run(reduced, eps, minPts);
  const clusters = new Map();

  dbscanResult.forEach((indices, clusterId) => {
    if (indices?.length) clusters.set(clusterId, indices);
  });

  // Fallback to K-means if DBSCAN fails
  if (clusters.size <= 1 && numRequirements > 5) {
    const k = computeAdaptiveK(reduced, numRequirements);
    const result = kmeans(reduced, k);
    clusters.clear();

    for (let i = 0; i < k; i++) {
      const clusterIndices = [];
      result.clusters.forEach((cid, idx) => {
        if (cid === i) clusterIndices.push(idx);
      });
      if (clusterIndices.length > 0) clusters.set(i, clusterIndices);
    }
  }

  console.log(`📊 Found ${clusters.size} clusters`);
  return clusters;
}