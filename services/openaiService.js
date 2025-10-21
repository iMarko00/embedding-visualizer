import OpenAI from "openai";
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/**
 * Generate human-readable cluster names using GPT
 */
export async function nameClusters(clusters, requirements, reduced) {
  const clusterMap = new Map();

  console.log("🏷️ Generating cluster names with OpenAI...");

  // Generate names for each cluster
  const clusterNames = await Promise.all(
    Array.from(clusters.entries()).map(async ([clusterId, indices]) => {
      const clusterRequirements = indices.map(i => requirements[i]);
      try {
        const completion = await client.chat.completions.create({
          model: "gpt-3.5-turbo",
          messages: [
            {
              role: "system",
              content:
                "You are an expert at analyzing software requirements and grouping them into meaningful categories. " +
                "Given a list of related requirements, provide a concise, descriptive name (2-4 words) that captures the common theme. " +
                "Examples: 'User Authentication', 'Payment Processing', 'Data Management', 'Security Features'.",
            },
            {
              role: "user",
              content:
                `Analyze these requirements and suggest a concise name for this cluster:\n\n` +
                clusterRequirements.map((r, i) => `${i + 1}. ${r}`).join("\n") +
                `\n\nProvide only the cluster name, nothing else.`,
            },
          ],
          max_tokens: 20,
          temperature: 0.3,
        });

        const name = completion.choices[0].message.content.trim().replace(/['"]/g, "");
        console.log(`✅ Cluster ${clusterId} named: "${name}"`);
        return { clusterId, name };
      } catch (error) {
        console.warn(`⚠️ Failed to name cluster ${clusterId}:`, error.message);
        return { clusterId, name: `Cluster ${clusterId}` };
      }
    })
  );

  // Map names for easier access
  const nameMap = new Map();
  clusterNames.forEach(({ clusterId, name }) => nameMap.set(clusterId, name));

  // Build labeled points and cluster metadata
  clusters.forEach((indices, clusterId) => {
    const clusterPoints = indices.map(i => reduced[i]);
    const cx = clusterPoints.reduce((a, b) => a + b[0], 0) / clusterPoints.length;
    const cy = clusterPoints.reduce((a, b) => a + b[1], 0) / clusterPoints.length;
    const radius = Math.max(...clusterPoints.map(([x, y]) => Math.hypot(x - cx, y - cy)));
    clusterMap.set(clusterId, {
      cx,
      cy,
      radius,
      name: nameMap.get(clusterId) || `Cluster ${clusterId}`,
    });
  });

  const labeledPoints = reduced.map(([x, y], i) => {
    let clusterId = -1;
    clusters.forEach((indices, id) => {
      if (indices.includes(i)) clusterId = id;
    });
    return { text: requirements[i], x, y, cluster: clusterId };
  });

  return {
    points: labeledPoints,
    clusters: Array.from(clusterMap.entries()).map(([id, { cx, cy, radius, name }]) => ({
      id,
      cx,
      cy,
      radius,
      name,
    })),
  };
}