import { useState, useEffect } from 'react';
import { ProductList } from './Components/ProductList';
import itemList from './Assets/random_products_175.json';
import './e-commerce-stylesheet.css';

type Product = {
  id: number;
  name: string;
  price: number;
  category: string;
  quantity: number;
  rating: number;
  image_link: string;
};

type BasketItem = {// the basket item type
  id: number;
  name: string;
  price: number;
  quantity: number;
};

function App() {
  const [searchTerm, setSearchTerm] = useState<string>('');// search term
  const [searchedProducts, setSearchedProducts] = useState<Product[]>(itemList);// search results
  const [basket, setBasket] = useState<BasketItem[]>([]);// basket
  const [sortOption, setSortOption] = useState<string>('AtoZ');// sort option
  const [inStockOnly, setInStockOnly] = useState<boolean>(false);// in stock only

  useEffect(() => {
    updateSearchedProducts();
  }, [searchTerm, sortOption, inStockOnly]);

  function showBasket() {
    let areaObject = document.getElementById('shopping-area');
    if (areaObject !== null) {
      areaObject.style.display = 'block';
    }
  }

  function hideBasket() {
    let areaObject = document.getElementById('shopping-area');
    if (areaObject !== null) {
      areaObject.style.display = 'none';
    }
  }

  function updateSearchedProducts() {
    let holderList = itemList.filter((product) =>
      product.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    if (inStockOnly) {
      holderList = holderList.filter(product => product.quantity > 0);
    }

    switch (sortOption) {// sort the products
      case 'AtoZ':
        holderList.sort((a, b) => a.name.localeCompare(b.name));
        break;
      case 'ZtoA':
        holderList.sort((a, b) => b.name.localeCompare(a.name));
        break;
      case '£LtoH':
        holderList.sort((a, b) => a.price - b.price);
        break;
      case '£HtoL':
        holderList.sort((a, b) => b.price - a.price);
        break;
      case '*LtoH':
        holderList.sort((a, b) => a.rating - b.rating);
        break;
      case '*HtoL':
        holderList.sort((a, b) => b.rating - a.rating);
        break;
      default:
        holderList.sort((a, b) => a.name.localeCompare(b.name));
    }

    setSearchedProducts(holderList);
  }

  function getSearchResultsText() {// get the search results text
    const count = searchedProducts.length;
    if (searchTerm === '') {
      return `${count} Product${count === 1 ? '' : 's'}`;
    } else if (count === 0) {
      return 'No search results found';
    } else {
      return `${count} Result${count === 1 ? '' : 's'}`;
    }
  }

  function addToBasket(productId: number) {// add to basket
    const product = itemList.find(p => p.id === productId);
    if (!product) return;

    const existingItem = basket.find(item => item.id === productId);// check if the item is already in the basket
    if (existingItem) {
      setBasket(basket.map(item =>
        item.id === productId ? { ...item, quantity: item.quantity + 1 } : item
      ));
    } else {// add the item to the basket
      const newItem: BasketItem = {
        id: product.id,
        name: product.name,
        price: product.price,
        quantity: 1
      };
      setBasket([...basket, newItem]);
    }
  }

  function removeFromBasket(productId: number) {// remove from basket
    const existingItem = basket.find(item => item.id === productId);
    if (!existingItem) return;

    if (existingItem.quantity > 1) {// reduce the quantity of the item
      setBasket(basket.map(item =>
        item.id === productId ? { ...item, quantity: item.quantity - 1 } : item
      ));
    } else {// remove the item from the basket
      setBasket(basket.filter(item => item.id !== productId));
    }
  }

  function getTotalCost() {// get the total cost of the basket
    return basket.reduce((total, item) => total + item.price * item.quantity, 0).toFixed(2);
  }

  return (
    <div id="container">
      <div id="logo-bar">
        <div id="logo-area">
          <img src="./src/assets/logo.png" alt="Logo"></img>
        </div>
        <div id="shopping-icon-area">
          <img id="shopping-icon" onClick={showBasket} src="./src/assets/shopping-basket.png" alt="Shopping Basket"></img>
        </div>
        <div id="shopping-area">
          <div id="exit-area">
            <p id="exit-icon" onClick={hideBasket}>x</p>
          </div>
          {basket.length === 0 ? (
            <p>Your basket is empty</p>
          ) : (
            <>
              {basket.map(item => (
                <div key={item.name} className="shopping-row">
                  <div className="shopping-information">
                    <p>{item.name} (£{item.price.toFixed(2)}) - {item.quantity}</p>
                  </div>
                  <button onClick={() => removeFromBasket(item.id)}>Remove</button>
                </div>
              ))}
              <p>Total: £{getTotalCost()}</p>
            </>
          )}
        </div>
      </div>
      <div id="search-bar">
        <input type="text" placeholder="Search..." onChange={e => setSearchTerm(e.target.value)}></input>
        <div id="control-area">
          <select onChange={e => setSortOption(e.target.value)} value={sortOption}>
            <option value="AtoZ">By name (A - Z)</option>
            <option value="ZtoA">By name (Z - A)</option>
            <option value="£LtoH">By price (low - high)</option>
            <option value="£HtoL">By price (high - low)</option>
            <option value="*LtoH">By rating (low - high)</option>
            <option value="*HtoL">By rating (high - low)</option>
          </select>
          <input id="inStock" type="checkbox" onChange={e => setInStockOnly(e.target.checked)}></input>
          <label htmlFor="inStock">In stock</label>
        </div>
      </div>
      <p id="results-indicator">{getSearchResultsText()}</p>
      <ProductList itemList={searchedProducts} onAddToBasket={addToBasket}/>
    </div>
  );
}

export default App;
