// ProductList.tsx

type ContentAreaProps = {
	itemList: Product[],
	onAddToBasket: (productId: number) => void
  }
  
  type Product = {// product type
	id: number,
	name: string,
	price: number,
	category: string,
	quantity: number,
	rating: number,
	image_link: string
  }
  
  export const ProductList = ({ itemList, onAddToBasket }: ContentAreaProps) => {
	return (
	  <div id="productList">
		{itemList.map((item) => (
		  <div key={item.id} className="product">
			<div className="product-top-bar">
			  <h2>{item.name}</h2>
			  <p>£{item.price.toFixed(2)} ({item.rating}/5)</p>
			</div>
			<img src={"./src/Assets/Product_Images/" + item.image_link} alt={item.name}></img>
			<button
			  disabled={item.quantity === 0}
			  onClick={() => onAddToBasket(item.id)}
			>
			  {item.quantity > 0 ? 'Add to basket' : 'Out of stock'}
			</button>
		  </div>
		))}
	  </div>
	);
  }
  