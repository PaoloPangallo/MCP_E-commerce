export type Feedback = {
  user: string;
  rating: string;
  comment: string;
  time: string;
};

export async function fetchSellerFeedback(
  seller: string
): Promise<Feedback[]> {

  const response = await fetch(`/seller/${seller}/feedback`);

  if (!response.ok) {
    throw new Error("Errore API");
  }

  const data = await response.json();
  return data.feedbacks || [];
}