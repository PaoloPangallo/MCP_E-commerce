export interface SearchItem {
  ebay_id: string
  title: string
  price: number
  currency: string
  image_url?: string
  url: string
  seller_name: string
  seller_rating?: number
  trust_score?: number
}